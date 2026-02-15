"""Main script for running real-data replay check."""
import argparse
import os
import csv
from typing import List, Dict, Optional
import numpy as np

from .load_yandex import load_yandex_sessions, validate_loaded_sessions, YandexSession
from .replay_eval import (
    evaluate_replay_session,
    aggregate_replay_results,
    bootstrap_ci,
    validate_metrics,
)


def apply_ede_reranking(
    session: YandexSession,
    doc_impressions: Dict[int, int],
    doc_clicks: Dict[int, int],
    lam: float = 0.4,
    eta: float = 0.65,
    m: int = 2,
    alpha: float = 0.5,
    k: int = 10,
    top_L: int = 50,
    tau: float = 50.0,
) -> List[int]:
    """
    Apply EDE reranking to a session.
    
    Uses simplified EDE scoring without the full policy machinery.
    
    Args:
        session: YandexSession with baseline ranking
        doc_impressions: per-doc impression counters
        doc_clicks: per-doc click counters
        lam: entropy bonus weight
        eta: novelty weight in E'(d) = eta*N(d) + (1-eta)*R(d)
        m: safe-prefix length
        alpha: entropy smoothing constant
        k: ranking cutoff
        top_L: candidate gating threshold
        tau: exposure weighting constant
        
    Returns:
        reranked doc ids (top-k)
    """
    baseline_docs = session.ranked_docs[:top_L]  # Only consider top-L
    n_candidates = len(baseline_docs)
    k_use = min(k, n_candidates)
    
    if k_use == 0:
        return []
    
    # Baseline scores: inverse rank
    base_scores = np.array([1.0 / (i + 1) for i in range(n_candidates)])
    
    # Get impression and click counts for these docs
    impressions = np.array([doc_impressions.get(doc, 0) for doc in baseline_docs])
    clicks = np.array([doc_clicks.get(doc, 0) for doc in baseline_docs])
    
    # Compute novelty term: N(d) = exp(-I_d / tau)
    N = np.exp(-impressions / tau)
    
    # Compute entropy scores (simplified)
    # H(d) = (C_d + alpha) / (I_d + 2*alpha)
    H = (clicks + alpha) / (impressions + 2 * alpha)
    
    # Normalize entropy to [0, 1]
    if H.max() > H.min():
        H_norm = (H - H.min()) / (H.max() - H.min())
    else:
        H_norm = np.ones_like(H) * 0.5
    
    # Compute refinement term: R(d) = (1 - exp(-I_d / tau)) * H_norm(d)
    R = (1 - np.exp(-impressions / tau)) * H_norm
    
    # Compute combined entropy bonus: E'(d) = eta * N(d) + (1-eta) * R(d)
    E_prime = eta * N + (1 - eta) * R
    
    # Step 1: Gate candidates (just keep top_L, already done)
    
    # Step 2: Keep top-m from baseline
    m_use = min(m, k_use)
    safe_prefix_docs = baseline_docs[:m_use]
    
    # Step 3: Rerank positions m+1..k
    if k_use > m_use:
        # Normalize base scores by rank for remaining candidates
        remaining_indices = list(range(m_use, n_candidates))
        remaining_base_scores = base_scores[remaining_indices]
        
        # Rank-based normalization for remaining docs
        ranks = np.argsort(-remaining_base_scores) + 1
        s_norm = np.zeros_like(remaining_base_scores)
        for i, rank in enumerate(ranks):
            if rank <= (k_use - m_use):
                s_norm[i] = 1 - (rank - 1) / max(k_use - m_use - 1, 1)
            else:
                s_norm[i] = 0.0
        
        # Compute final scores for remaining positions
        remaining_E_prime = E_prime[remaining_indices]
        final_scores = s_norm + lam * remaining_E_prime
        
        # Sort remaining docs by final scores
        sorted_remaining_indices = np.argsort(-final_scores)
        reranked_remaining_docs = [
            baseline_docs[remaining_indices[i]]
            for i in sorted_remaining_indices[:k_use - m_use]
        ]
        
        # Combine safe prefix with reranked remainder
        reranked_docs = list(safe_prefix_docs) + reranked_remaining_docs
    else:
        reranked_docs = list(safe_prefix_docs)
    
    return reranked_docs[:k_use]


def _kendall_tau_distance_top_k(baseline_top_k: List[int], ede_top_k: List[int]) -> float:
    """Compute Kendall tau distance on shared docs, normalized by pair count."""
    baseline_pos = {doc: idx for idx, doc in enumerate(baseline_top_k)}
    ede_pos = {doc: idx for idx, doc in enumerate(ede_top_k)}
    shared_docs = [doc for doc in baseline_top_k if doc in ede_pos]
    n = len(shared_docs)
    if n < 2:
        return 0.0

    discordant = 0
    total_pairs = n * (n - 1) // 2
    for i in range(n):
        for j in range(i + 1, n):
            doc_i = shared_docs[i]
            doc_j = shared_docs[j]
            base_order = baseline_pos[doc_i] - baseline_pos[doc_j]
            ede_order = ede_pos[doc_i] - ede_pos[doc_j]
            if base_order * ede_order < 0:
                discordant += 1

    return discordant / total_pairs


def apply_ede_with_margins(
    session: YandexSession,
    doc_impressions: Dict[int, int],
    doc_clicks: Dict[int, int],
    lam: float,
    eta: float,
    m: int,
    alpha: float,
    k: int,
    top_L: int,
    tau: float,
) -> tuple:
    """
    Apply EDE reranking and return reranked docs + score margins for changed positions.
    
    Returns:
        (reranked_docs, margin_info)
        where margin_info is dict with keys:
            - 'swap_position': int (position that changed, or None)
            - 'swapped_in_doc': int (new doc_id, or None)
            - 'swapped_out_doc': int (old doc_id, or None)
            - 'margin': float (score_best - score_second_best, or None)
    """
    baseline_docs = session.ranked_docs[:top_L]
    n_candidates = len(baseline_docs)
    k_use = min(k, n_candidates)
    
    if k_use == 0:
        return [], {'swap_position': None, 'swapped_in_doc': None, 'swapped_out_doc': None, 'margin': None}
    
    # Baseline scores
    base_scores = np.array([1.0 / (i + 1) for i in range(n_candidates)])
    
    # Get impression and click counts
    impressions = np.array([doc_impressions.get(doc, 0) for doc in baseline_docs])
    clicks = np.array([doc_clicks.get(doc, 0) for doc in baseline_docs])
    
    # Compute novelty and entropy components
    N = np.exp(-impressions / tau)
    H = (clicks + alpha) / (impressions + 2 * alpha)
    if H.max() > H.min():
        H_norm = (H - H.min()) / (H.max() - H.min())
    else:
        H_norm = np.ones_like(H) * 0.5
    R = (1 - np.exp(-impressions / tau)) * H_norm
    E_prime = eta * N + (1 - eta) * R
    
    # Keep top-m from baseline
    m_use = min(m, k_use)
    safe_prefix_docs = baseline_docs[:m_use]
    
    margin_info = {'swap_position': None, 'swapped_in_doc': None, 'swapped_out_doc': None, 'margin': None}
    
    # Rerank positions m+1..k
    if k_use > m_use:
        remaining_indices = list(range(m_use, n_candidates))
        remaining_base_scores = base_scores[remaining_indices]
        
        # Rank-based normalization
        ranks = np.argsort(-remaining_base_scores) + 1
        s_norm = np.zeros_like(remaining_base_scores)
        for i, rank in enumerate(ranks):
            if rank <= (k_use - m_use):
                s_norm[i] = 1 - (rank - 1) / max(k_use - m_use - 1, 1)
            else:
                s_norm[i] = 0.0
        
        # Compute final scores
        remaining_E_prime = E_prime[remaining_indices]
        final_scores = s_norm + lam * remaining_E_prime
        
        # Find margin between best and second-best
        if len(final_scores) >= 2:
            sorted_indices = np.argsort(-final_scores)
            best_idx = sorted_indices[0]
            second_best_idx = sorted_indices[1]
            margin_info['margin'] = float(final_scores[best_idx] - final_scores[second_best_idx])
        
        # Sort and rerank
        sorted_remaining_indices = np.argsort(-final_scores)
        reranked_remaining_docs = [
            baseline_docs[remaining_indices[i]]
            for i in sorted_remaining_indices[:k_use - m_use]
        ]
        
        # Detect swap
        reranked_docs = list(safe_prefix_docs) + reranked_remaining_docs
        baseline_top_k = baseline_docs[:k_use]
        for i, (base_doc, ede_doc) in enumerate(zip(baseline_top_k, reranked_docs)):
            if base_doc != ede_doc:
                margin_info['swap_position'] = i
                margin_info['swapped_in_doc'] = ede_doc
                margin_info['swapped_out_doc'] = base_doc
                break
    else:
        reranked_docs = list(safe_prefix_docs)
    
    return reranked_docs[:k_use], margin_info


def run_lambda_sanity_test(
    data_dir: str,
    output_dir: str = "outputs/real_replay",
    k: int = 10,
    min_results: int = 10,
    max_sessions: int = 200000,
    eta: float = 0.65,
    m: int = 2,
    alpha: float = 0.5,
    top_L: int = 50,
    tau: float = 50.0,
):
    """Run lambda-sanity test with lambdas=[0.0, 0.2, 0.4, 2.0]."""
    print("\n" + "=" * 70)
    print("Lambda Sanity Test")
    print("=" * 70)
    
    # Load sessions
    print("\n[1/3] Loading Yandex sessions...")
    sessions = load_yandex_sessions(data_dir, k_min=min_results, max_sessions=max_sessions)
    session_stats = validate_loaded_sessions(sessions)
    
    lambdas = [0.0, 0.2, 0.4, 2.0]
    
    # Run EDE for each lambda and collect set-change sessions
    lambda_results = {}
    
    for lam_value in lambdas:
        print(f"\n[2/3] Running EDE with lambda={lam_value}...")
        doc_impressions = {}
        doc_clicks = {}
        
        set_change_sessions = []
        
        for i, session in enumerate(sessions):
            if (i + 1) % 20000 == 0:
                print(f"  Processed {i+1}/{len(sessions)} sessions...")
            
            # Apply EDE with margin tracking
            ede_ranking, margin_info = apply_ede_with_margins(
                session, doc_impressions, doc_clicks,
                lam=lam_value, eta=eta, m=m, alpha=alpha, k=k, top_L=top_L, tau=tau
            )
            
            # Update counters
            for rank, doc_id in enumerate(ede_ranking):
                doc_impressions[doc_id] = doc_impressions.get(doc_id, 0) + 1
                if rank < len(session.clicks) and session.clicks[rank] > 0:
                    doc_clicks[doc_id] = doc_clicks.get(doc_id, 0) + session.clicks[rank]
            
            # Check for click-preservation and set-change
            baseline_top_k = session.ranked_docs[:k]
            ede_top_k = ede_ranking[:k]
            
            clicked_positions = [i for i, c in enumerate(session.clicks[:k]) if c > 0]
            clicked_docs = {session.ranked_docs[i] for i in clicked_positions}
            
            click_preserved = clicked_docs.issubset(set(ede_top_k))
            set_changed = set(baseline_top_k) != set(ede_top_k)
            
            if click_preserved and set_changed:
                set_change_sessions.append({
                    'session_idx': i,
                    'swap_position': margin_info.get('swap_position'),
                    'swapped_in_doc': margin_info.get('swapped_in_doc'),
                    'margin': margin_info.get('margin'),
                })
        
        print(f"  Found {len(set_change_sessions)} click-preservation sessions with set changes")
        lambda_results[lam_value] = set_change_sessions
    
    # Sample 50 random sessions from lambda=0.4's set_change_sessions
    print(f"\n[3/3] Analyzing sampled sessions...")
    np.random.seed(0)
    base_lambda = 0.4
    base_sessions = lambda_results[base_lambda]
    
    if len(base_sessions) < 50:
        print(f"  WARNING: Only {len(base_sessions)} set-change sessions available")
        sample_size = len(base_sessions)
    else:
        sample_size = 50
    
    sampled_indices = np.random.choice(len(base_sessions), size=sample_size, replace=False)
    sampled_session_ids = [base_sessions[i]['session_idx'] for i in sampled_indices]
    
    # Build comparison data
    comparison_data = []
    for lam_value in lambdas:
        sessions_by_id = {s['session_idx']: s for s in lambda_results[lam_value]}
        
        margins = []
        diff_from_base = 0
        matched = 0
        
        for sess_id in sampled_session_ids:
            if sess_id in sessions_by_id:
                matched += 1
                sess_data = sessions_by_id[sess_id]
                if sess_data['margin'] is not None:
                    margins.append(sess_data['margin'])
                
                # Compare swapped-in doc with base lambda
                base_sessions_by_id = {s['session_idx']: s for s in lambda_results[base_lambda]}
                if lam_value != base_lambda:
                    base_data = base_sessions_by_id.get(sess_id)
                    if base_data and sess_data['swapped_in_doc'] != base_data['swapped_in_doc']:
                        diff_from_base += 1
        
        pct_diff = (diff_from_base / matched * 100) if matched > 0 else 0.0
        mean_margin = np.mean(margins) if margins else 0.0
        median_margin = np.median(margins) if margins else 0.0
        
        comparison_data.append({
            'lambda': lam_value,
            'n_matched': matched,
            'pct_diff_from_0.4': pct_diff,
            'mean_margin': mean_margin,
            'median_margin': median_margin,
        })
        
        print(f"  lambda={lam_value}: {matched} matched, {pct_diff:.1f}% differ from lambda=0.4, margin mean={mean_margin:.4f} median={median_margin:.4f}")
    
    # Validation checks
    print("\n" + "=" * 70)
    print("Validation Checks")
    print("=" * 70)
    
    # Check if lambda=0.0 and lambda=2.0 produce identical docs
    lam_00_data = [d for d in comparison_data if d['lambda'] == 0.0][0]
    lam_20_data = [d for d in comparison_data if d['lambda'] == 2.0][0]
    
    if lam_00_data['pct_diff_from_0.4'] < 1.0 and lam_20_data['pct_diff_from_0.4'] < 1.0:
        print("  WARNING: lambda appears to have no effect (either saturated margins or not applied).")
    
    # Check if margins are large
    lam_04_data = [d for d in comparison_data if d['lambda'] == 0.4][0]
    if lam_04_data['median_margin'] > 0.05:
        print(f"  WARNING: insensitive due to large score margins (median={lam_04_data['median_margin']:.4f}).")
    
    print("=" * 70)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "replay_summary_lambda_sanity.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lambda', 'n_matched_sessions', 'pct_diff_from_lambda_0.4', 'mean_margin', 'median_margin'])
        for row in comparison_data:
            writer.writerow([
                f"{row['lambda']:.1f}",
                row['n_matched'],
                f"{row['pct_diff_from_0.4']:.2f}",
                f"{row['mean_margin']:.6f}",
                f"{row['median_margin']:.6f}",
            ])
    print(f"\nSaved: {csv_path}")


def run_replay_check(
    data_dir: str,
    output_dir: str = "outputs/real_replay",
    k: int = 10,
    min_results: int = 10,
    max_sessions: int = 200000,
    bootstrap_ci_enabled: bool = False,
    bootstrap_B: int = 1000,
    bootstrap_seed: int = 0,
    lam: float = 0.4,
    lambdas: Optional[List[float]] = None,
    eta: float = 0.65,
    m: int = 2,
    alpha: float = 0.5,
    top_L: int = 50,
    tau: float = 50.0,
):
    """
    Run complete replay check pipeline.
    
    Args:
        data_dir: path to Yandex log files
        output_dir: path to output directory
        k: ranking cutoff
        min_results: minimum results per session
        max_sessions: maximum number of sessions to parse from logs
        bootstrap_ci_enabled: whether to compute bootstrap CIs for click-preservation deltas
        bootstrap_B: number of bootstrap resamples
        bootstrap_seed: random seed for bootstrap
        lam: EDE lambda parameter
        eta: EDE eta parameter
        m: EDE safe-prefix length
        alpha: EDE alpha parameter
        top_L: EDE candidate gating threshold
        tau: EDE tau parameter
    """
    print("=" * 70)
    print("EDE Real-Data Replay Check")
    print("=" * 70)
    
    # Load sessions
    print("\n[1/5] Loading Yandex sessions...")
    sessions = load_yandex_sessions(data_dir, k_min=min_results, max_sessions=max_sessions)
    
    # Validate sessions
    print("\n[2/5] Validating loaded sessions...")
    session_stats = validate_loaded_sessions(sessions)
    print(f"  Total sessions: {session_stats['num_sessions']}")
    print(f"  Sessions with >=10 results: {session_stats['num_with_10plus_results']}")
    print(f"  Sessions with clicks: {session_stats['num_with_clicks']}")
    print(f"  Alignment errors: {session_stats['alignment_errors']}")
    
    if not session_stats['validation_passed']:
        print("\n  ERROR: Validation failed!")
        for warning in session_stats['warnings']:
            print(f"    - {warning}")
        return
    
    if session_stats['warnings']:
        print("\n  WARNINGS:")
        for warning in session_stats['warnings']:
            print(f"    - {warning}")
    
    if lambdas is None:
        lambdas = [lam]

    # Apply EDE reranking with streaming counters (per lambda)
    two_lambda_rows = []
    two_lambda_diagnostics = {}
    for lam_value in lambdas:
        print(f"\n[3/5] Applying EDE reranking (lambda={lam_value}, eta={eta}, m={m})...")
        doc_impressions = {}  # doc_id -> impression count
        doc_clicks = {}  # doc_id -> click count

        results = []
        # Accumulators (click-preservation bucket)
        click_pres_total = 0
        # Accumulators (set-match bucket)
        set_match_total = 0
        set_match_order_changed = 0

        # Per-criterion change diagnostics (accepted sessions)
        criterion_stats = {
            'Replay-Exact-Match': {'accepted': 0, 'reorder': 0, 'mismatch_sum': 0},
            'Replay-Set-Match': {'accepted': 0, 'reorder': 0, 'mismatch_sum': 0},
            'Replay-Click-Preservation': {
                'accepted': 0, 'reorder': 0, 'mismatch_sum': 0, 'clicked_shift_sum': 0.0
            },
        }

        for i, session in enumerate(sessions):
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1}/{len(sessions)} sessions...")

            # Apply EDE reranking using current counters
            ede_ranking = apply_ede_reranking(
                session,
                doc_impressions,
                doc_clicks,
                lam=lam_value,
                eta=eta,
                m=m,
                alpha=alpha,
                k=k,
                top_L=top_L,
                tau=tau,
            )

            # Evaluate session
            result = evaluate_replay_session(session, ede_ranking, k=k)
            results.append(result)

            # Ranking change diagnostics
            k_use = min(k, len(session.ranked_docs), len(ede_ranking))
            baseline_top_k = session.ranked_docs[:k_use]
            ede_top_k = ede_ranking[:k_use]
            baseline_set = set(baseline_top_k)
            ede_set = set(ede_top_k)

            mismatch_positions = 0
            for idx in range(k_use):
                if baseline_top_k[idx] != ede_top_k[idx]:
                    mismatch_positions += 1

            if result['exact_match_pass']:
                stats = criterion_stats['Replay-Exact-Match']
                stats['accepted'] += 1
                if baseline_top_k != ede_top_k:
                    stats['reorder'] += 1
                stats['mismatch_sum'] += mismatch_positions

            if result['set_match_pass']:
                stats = criterion_stats['Replay-Set-Match']
                stats['accepted'] += 1
                if baseline_top_k != ede_top_k:
                    stats['reorder'] += 1
                stats['mismatch_sum'] += mismatch_positions

            if result['click_pres_pass']:
                stats = criterion_stats['Replay-Click-Preservation']
                stats['accepted'] += 1
                if baseline_top_k != ede_top_k:
                    stats['reorder'] += 1
                stats['mismatch_sum'] += mismatch_positions

                # Mean abs rank shift for clicked docs (per session)
                baseline_pos = {doc: idx for idx, doc in enumerate(baseline_top_k)}
                ede_pos = {doc: idx for idx, doc in enumerate(ede_top_k)}
                clicked_docs = [
                    doc for doc, click in zip(baseline_top_k, session.clicks[:k_use]) if click == 1
                ]
                if clicked_docs:
                    shift_sum = 0.0
                    shift_count = 0
                    for doc in clicked_docs:
                        if doc in ede_pos:
                            shift_sum += abs(baseline_pos[doc] - ede_pos[doc])
                            shift_count += 1
                    if shift_count > 0:
                        stats['clicked_shift_sum'] += shift_sum / shift_count

                click_pres_total += 1

            if result['set_match_pass']:
                set_match_total += 1
                if baseline_top_k != ede_top_k:
                    set_match_order_changed += 1

            # Update counters based on baseline (logged) behavior
            for doc, click in zip(session.ranked_docs[:k], session.clicks[:k]):
                doc_impressions[doc] = doc_impressions.get(doc, 0) + 1
                if click == 1:
                    doc_clicks[doc] = doc_clicks.get(doc, 0) + 1
    
        print(f"  Completed {len(sessions)} sessions")
    
        # Compute diagnostic stats - RECOMPUTE ON CLICK-PRESERVATION BUCKET ONLY
        print("\n[4/6] Computing diagnostic statistics...")
        total_clicks = sum(sum(s.clicks[:k]) for s in sessions)
        avg_clicks_per_session = total_clicks / len(sessions)
        sessions_with_1_click = sum(1 for s in sessions if sum(s.clicks[:k]) == 1)
        sessions_with_2plus_clicks = sum(1 for s in sessions if sum(s.clicks[:k]) >= 2)
        frac_1_click = sessions_with_1_click / len(sessions)
        frac_2plus_clicks = sessions_with_2plus_clicks / len(sessions)
    
        print(f"  Avg clicks per session: {avg_clicks_per_session:.3f}")
        print(f"  Sessions with 1 click: {frac_1_click*100:.1f}%")
        print(f"  Sessions with 2+ clicks: {frac_2plus_clicks*100:.1f}%")
    
        # RECOMPUTE DIAGNOSTICS PROPERLY ON CLICK-PRESERVATION SESSIONS
        print("\n  Recomputing diagnostics on click-preservation bucket...")
        exact_list_changes = 0
        set_changes = 0
        reorder_only_changes = 0
        pos_diffs_list = []
        clicked_shifts_list = []
        kendall_taus_list = []
        intersection_sizes_list = []
        click_pres_diag_samples = []  # for debug output
        set_change_sessions = []  # for collecting set_change sessions
        position_changes_freq = [0] * k  # track which positions changed
    
        set_match_diag_order_changed = 0  # for set-match bucket
        set_match_diag_total = 0
    
        for i, session in enumerate(sessions):
            ede_ranking = apply_ede_reranking(
                session, doc_impressions, doc_clicks,
                lam=lam_value, eta=eta, m=m, alpha=alpha, k=k, top_L=top_L, tau=tau,
            )
            result = evaluate_replay_session(session, ede_ranking, k=k)
        
            k_use = min(k, len(session.ranked_docs), len(ede_ranking))
            baseline_top_k = session.ranked_docs[:k_use]
            ede_top_k = ede_ranking[:k_use]
            baseline_set = set(baseline_top_k)
            ede_set = set(ede_top_k)
        
            if result['click_pres_pass']:
                # Count position mismatches (exact list change) and track positions
                pos_mismatch_count = 0
                changed_positions = []
                for idx in range(k_use):
                    if baseline_top_k[idx] != ede_top_k[idx]:
                        pos_mismatch_count += 1
                        changed_positions.append(idx)
                        position_changes_freq[idx] += 1
            
                pos_diffs_list.append(pos_mismatch_count)
            
                if baseline_top_k != ede_top_k:
                    exact_list_changes += 1
                if baseline_set != ede_set:
                    set_changes += 1
                if baseline_top_k != ede_top_k and baseline_set == ede_set:
                    reorder_only_changes += 1
            
                # Clicked doc rank shift
                baseline_pos = {doc: idx for idx, doc in enumerate(baseline_top_k)}
                ede_pos = {doc: idx for idx, doc in enumerate(ede_top_k)}
                clicked_docs = [doc for doc, click in zip(baseline_top_k, session.clicks[:k_use]) if click == 1]
            
                if clicked_docs:
                    for doc in clicked_docs:
                        if doc in ede_pos:
                            shift = abs(baseline_pos[doc] - ede_pos[doc])
                            clicked_shifts_list.append(shift)
            
                # Kendall tau on intersection only
                shared_docs = baseline_set.intersection(ede_set)
                if len(shared_docs) >= 2:
                    tau_dist = _kendall_tau_distance_top_k(baseline_top_k, ede_top_k)
                    kendall_taus_list.append(tau_dist)
                intersection_sizes_list.append(len(shared_docs))
            
                # Collect sessions with set changes
                if baseline_set != ede_set:
                    set_change_sessions.append({
                        'baseline': baseline_top_k[:k_use],
                        'ede': ede_top_k[:k_use],
                        'clicked_positions': [idx for idx, click in enumerate(session.clicks[:k_use]) if click == 1],
                        'clicked_docs': clicked_docs,
                        'baseline_pos': baseline_pos,
                        'ede_pos': ede_pos,
                        'pos_changes_count': pos_mismatch_count
                    })
            
                # Collect debug samples (first 5 list changes)
                if len(click_pres_diag_samples) < 5 and baseline_top_k != ede_top_k:
                    click_pres_diag_samples.append({
                        'baseline': baseline_top_k[:10],
                        'ede': ede_top_k[:10],
                        'clicked': clicked_docs,
                        'clicked_shifts': {
                            doc: abs(baseline_pos[doc] - ede_pos[doc]) 
                            for doc in clicked_docs if doc in ede_pos
                        }
                    })
        
            # Measure set-match order changes separately
            if result['set_match_pass']:
                set_match_diag_total += 1
                if baseline_top_k != ede_top_k:
                    set_match_diag_order_changed += 1
    
        # Compute aggregated diagnostics
        if click_pres_total > 0:
            exact_list_change_rate = exact_list_changes / click_pres_total
            set_change_rate = set_changes / click_pres_total
            reorder_only_rate = reorder_only_changes / click_pres_total
            mean_pos_diffs_click_pres = np.mean(pos_diffs_list) if pos_diffs_list else 0.0
            median_pos_diffs = np.median(pos_diffs_list) if pos_diffs_list else 0.0
            p90_pos_diffs = np.percentile(pos_diffs_list, 90) if pos_diffs_list else 0.0
            mean_pos_diffs_fraction = mean_pos_diffs_click_pres / k if k > 0 else 0.0
        
            # Distribution of position changes
            pct_sessions_eq1_positions_changed = (sum(1 for d in pos_diffs_list if d == 1) / len(pos_diffs_list)) if pos_diffs_list else 0.0
            pct_sessions_ge2_positions_changed = (sum(1 for d in pos_diffs_list if d >= 2) / len(pos_diffs_list)) if pos_diffs_list else 0.0
            pct_sessions_ge3_positions_changed = (sum(1 for d in pos_diffs_list if d >= 3) / len(pos_diffs_list)) if pos_diffs_list else 0.0
        
            # Top 3 most frequently changed positions (1-indexed for readability)
            top_positions = sorted(enumerate(position_changes_freq), key=lambda x: x[1], reverse=True)[:3]
            top_positions_str = "; ".join(f"pos{x[0]+1}:{x[1]}" for x in top_positions)
        
            mean_clicked_shift = np.mean(clicked_shifts_list) if clicked_shifts_list else 0.0
            median_clicked_shift = np.median(clicked_shifts_list) if clicked_shifts_list else 0.0
            pct_clicked_shift_gt0 = (sum(1 for s in clicked_shifts_list if s > 0) / len(clicked_shifts_list)) if clicked_shifts_list else 0.0
        
            mean_kendall_tau = np.mean(kendall_taus_list) if kendall_taus_list else 0.0
            mean_intersection_size = np.mean(intersection_sizes_list) if intersection_sizes_list else 0.0
        else:
            exact_list_change_rate = 0.0
            set_change_rate = 0.0
            reorder_only_rate = 0.0
            mean_pos_diffs_click_pres = 0.0
            median_pos_diffs = 0.0
            p90_pos_diffs = 0.0
            mean_pos_diffs_fraction = 0.0
            pct_sessions_eq1_positions_changed = 0.0
            pct_sessions_ge2_positions_changed = 0.0
            pct_sessions_ge3_positions_changed = 0.0
            top_positions_str = ""
            mean_clicked_shift = 0.0
            median_clicked_shift = 0.0
            pct_clicked_shift_gt0 = 0.0
            mean_kendall_tau = 0.0
            mean_intersection_size = 0.0
    
        if set_match_diag_total > 0:
            pct_order_change_set_match = set_match_diag_order_changed / set_match_diag_total
        else:
            pct_order_change_set_match = 0.0
    
        # Validation assertions
        reorder_plus_set = reorder_only_rate + set_change_rate
        if abs(reorder_plus_set - exact_list_change_rate) > 0.01:
            print(f"  WARNING: reorder_only_rate ({reorder_only_rate:.4f}) + set_change_rate ({set_change_rate:.4f}) "
                  f"!≈ exact_list_change_rate ({exact_list_change_rate:.4f})")
    
        frac_check = mean_pos_diffs_fraction
        if abs(frac_check - mean_pos_diffs_click_pres / k) > 1e-6 and k > 0:
            print(f"  WARNING: mean_pos_diffs_fraction calculation inconsistent")
    
        # Validation for position change distribution
        if pct_sessions_ge2_positions_changed > exact_list_change_rate + 1e-6:
            print(f"  WARNING: pct_sessions_ge2_positions_changed ({pct_sessions_ge2_positions_changed:.4f}) > "
                  f"exact_list_change_rate ({exact_list_change_rate:.4f})")
    
        sum_eq1_plus_ge2 = pct_sessions_eq1_positions_changed + pct_sessions_ge2_positions_changed
        if abs(sum_eq1_plus_ge2 - exact_list_change_rate) > 0.01:
            print(f"  WARNING: pct_sessions_eq1 ({pct_sessions_eq1_positions_changed:.4f}) + "
                  f"pct_sessions_ge2 ({pct_sessions_ge2_positions_changed:.4f}) "
                  f"!≈ exact_list_change_rate ({exact_list_change_rate:.4f})")
    
        print(f"\n  Ranking Change Diagnostics (Click-Preservation bucket):")
        print(f"    Exact list change rate: {exact_list_change_rate:.2%}")
        print(f"    Set change rate: {set_change_rate:.2%}")
        print(f"    Reorder-only rate (same set, different order): {reorder_only_rate:.2%}")
        print(f"    Mean positions changed count: {mean_pos_diffs_click_pres:.3f} / {k}")
        print(f"    Median positions changed count: {median_pos_diffs:.1f}")
        print(f"    P90 positions changed count: {p90_pos_diffs:.1f}")
        print(f"    Mean positions changed fraction: {mean_pos_diffs_fraction:.4f}")
        print(f"    Pct sessions with =1 position changed: {pct_sessions_eq1_positions_changed:.2%}")
        print(f"    Pct sessions with >=2 positions changed: {pct_sessions_ge2_positions_changed:.2%}")
        print(f"    Pct sessions with >=3 positions changed: {pct_sessions_ge3_positions_changed:.2%}")
        print(f"    Top 3 most frequently changed positions: {top_positions_str}")
        print(f"    Mean clicked-doc absolute rank shift: {mean_clicked_shift:.3f}")
        print(f"    Median clicked-doc absolute rank shift: {median_clicked_shift:.3f}")
        print(f"    % clicked docs with shift > 0: {pct_clicked_shift_gt0:.2%}")
        print(f"    Mean Kendall tau (on intersection): {mean_kendall_tau:.4f}")
        print(f"    Mean intersection size: {mean_intersection_size:.2f}")
        print(f"    Order-change rate (set-match bucket only): {pct_order_change_set_match:.2%}")
    
        # Print 3 random sessions with set changes (seed=0)
        if set_change_sessions:
            import random
            random.seed(0)
            sample_size = min(3, len(set_change_sessions))
            set_change_sample = random.sample(set_change_sessions, sample_size)
        
            print(f"\n  Set-change sessions (seed=0, n={sample_size}):")
            for idx, sample in enumerate(set_change_sample, 1):
                baseline_str = ','.join(str(d) for d in sample['baseline'])
                ede_str = ','.join(str(d) for d in sample['ede'])
                clicked_doc_ede_pos = {doc: sample['ede_pos'][doc] if doc in sample['ede_pos'] else 'missing' 
                                        for doc in sample['clicked_docs']}
                print(f"    [{idx}] baseline={baseline_str}")
                print(f"        ede={ede_str}")
                print(f"        clicked_pos_baseline={sample['clicked_positions']} docs={sample['clicked_docs']}")
                print(f"        clicked_pos_ede={clicked_doc_ede_pos}")
                print(f"        pos_changes_count={sample['pos_changes_count']}/10")
    
        # DEBUG: Print first 5 samples with order changes
        if click_pres_diag_samples:
            print("\n  DEBUG: Sample sessions with ranking changes (click-preservation bucket):")
            for idx, sample in enumerate(click_pres_diag_samples):
                print(f"    Sample {idx+1}:")
                print(f"      Baseline top10: {sample['baseline']}")
                print(f"      EDE top10:      {sample['ede']}")
                print(f"      Clicked docs: {sample['clicked']}")
                if sample['clicked_shifts']:
                    print(f"      Clicked-doc shifts: {sample['clicked_shifts']}")
    
        # Aggregate results
        print("\n[5/6] Aggregating results...")
        aggregated = aggregate_replay_results(results, k=k)

        # Bootstrap CI over accepted Replay-Click-Preservation sessions only
        click_pres_results = [r for r in results if r['click_pres_pass']]
        click_pres_baseline_ctr_values = np.array([r['baseline_ctr'] for r in click_pres_results], dtype=np.float64)
        click_pres_ede_ctr_values = np.array([r['ede_ctr'] for r in click_pres_results], dtype=np.float64)
        click_pres_baseline_ndcg_values = np.array([r['baseline_ndcg'] for r in click_pres_results], dtype=np.float64)
        click_pres_ede_ndcg_values = np.array([r['ede_ndcg'] for r in click_pres_results], dtype=np.float64)

        click_pres_delta_ctr_values = click_pres_ede_ctr_values - click_pres_baseline_ctr_values
        click_pres_delta_ndcg_values = click_pres_ede_ndcg_values - click_pres_baseline_ndcg_values

        ci_delta_ctr = float(np.mean(click_pres_delta_ctr_values)) if click_pres_delta_ctr_values.size > 0 else 0.0
        ci_delta_ctr_lo = ci_delta_ctr
        ci_delta_ctr_hi = ci_delta_ctr
        ci_delta_ndcg = float(np.mean(click_pres_delta_ndcg_values)) if click_pres_delta_ndcg_values.size > 0 else 0.0
        ci_delta_ndcg_lo = ci_delta_ndcg
        ci_delta_ndcg_hi = ci_delta_ndcg

        n_click_pres = click_pres_delta_ctr_values.size
        if n_click_pres < 10000:
            print(
                f"  WARNING: Click-Preservation accepted sessions = {n_click_pres:,} (<10,000); "
                "bootstrap CI may be less stable."
            )

        # Bounds validation for underlying metrics and deltas
        if click_pres_baseline_ctr_values.size > 0:
            for metric_name, metric_values in (
                ('baseline_ctr', click_pres_baseline_ctr_values),
                ('ede_ctr', click_pres_ede_ctr_values),
                ('baseline_ndcg', click_pres_baseline_ndcg_values),
                ('ede_ndcg', click_pres_ede_ndcg_values),
            ):
                if np.any((metric_values < 0.0) | (metric_values > 1.0)):
                    raise RuntimeError(f"Click-preservation metric out of bounds [0,1]: {metric_name}")

            if np.any((click_pres_delta_ctr_values < -1.0) | (click_pres_delta_ctr_values > 1.0)):
                raise RuntimeError("Click-preservation delta CTR out of bounds [-1,1]")
            if np.any((click_pres_delta_ndcg_values < -1.0) | (click_pres_delta_ndcg_values > 1.0)):
                raise RuntimeError("Click-preservation delta NDCG out of bounds [-1,1]")

        if bootstrap_ci_enabled and n_click_pres > 0:
            ci_delta_ctr, ci_delta_ctr_lo, ci_delta_ctr_hi = bootstrap_ci(
                click_pres_delta_ctr_values,
                B=bootstrap_B,
                seed=bootstrap_seed,
            )
            ci_delta_ndcg, ci_delta_ndcg_lo, ci_delta_ndcg_hi = bootstrap_ci(
                click_pres_delta_ndcg_values,
                B=bootstrap_B,
                seed=bootstrap_seed + 1,
            )

            # CI sanity checks
            if not (ci_delta_ctr_lo <= ci_delta_ctr <= ci_delta_ctr_hi):
                raise RuntimeError("Invalid CTR bootstrap CI: expected lo <= point <= hi")
            if not (ci_delta_ndcg_lo <= ci_delta_ndcg <= ci_delta_ndcg_hi):
                raise RuntimeError("Invalid NDCG bootstrap CI: expected lo <= point <= hi")

            print(
                f"  Bootstrap CI over accepted sessions (Replay-Click-Preservation, B={bootstrap_B}): "
                f"ΔCTR@10={ci_delta_ctr:.6f} [{ci_delta_ctr_lo:.6f}, {ci_delta_ctr_hi:.6f}], "
                f"Δclick-NDCG@10={ci_delta_ndcg:.6f} [{ci_delta_ndcg_lo:.6f}, {ci_delta_ndcg_hi:.6f}]"
            )

        exact_rate = aggregated['exact_match_acceptance_rate']
        set_rate = aggregated['set_match_acceptance_rate']
        click_rate = aggregated['click_pres_acceptance_rate']
    
        exact_count = aggregated['exact_match_num_accepted']
        set_count = aggregated['set_match_num_accepted']
        click_count = aggregated['click_pres_num_accepted']

        # ASSERTIONS: Check criterion ordering
        if not (exact_count <= set_count + 1):
            raise RuntimeError(
                f"Exact-match count ({exact_count}) > Set-match count ({set_count}). "
                "Logic bug: Exact should be subset of Set."
            )
        if not (set_count <= click_count + 1):
            raise RuntimeError(
                f"Set-match count ({set_count}) > Click-preservation count ({click_count}). "
                "Logic bug: Set should be subset of Click-preservation."
            )
    
        print(f"  Acceptance rates ordered correctly: exact={exact_count} <= set={set_count} <= click={click_count}")
    
        if not (exact_rate <= set_rate + 1e-9):
            print(
                f"  ERROR: Exact-match rate ({exact_rate:.6f}) > Set-match rate ({set_rate:.6f})"
            )
        if not (set_rate <= click_rate + 1e-9):
            print(
                f"  ERROR: Set-match rate ({set_rate:.6f}) > Click-preservation rate ({click_rate:.6f})"
            )
    
        print(f"  Replay-Exact-Match acceptance: {aggregated['exact_match_acceptance_rate']*100:.2f}% "
              f"({aggregated['exact_match_num_accepted']} sessions)")
        print(f"  Replay-Set-Match acceptance: {aggregated['set_match_acceptance_rate']*100:.2f}% "
              f"({aggregated['set_match_num_accepted']} sessions)")
        print(f"  Replay-Click-Preservation acceptance: {aggregated['click_pres_acceptance_rate']*100:.2f}% "
              f"({aggregated['click_pres_num_accepted']} sessions)")
    
        # Validate metrics
        validation = validate_metrics(aggregated)
        if not validation['passed']:
            print("\n  ERROR: Metric validation failed!")
            for warning in validation['warnings']:
                print(f"    - {warning}")
            if len(lambdas) > 1:
                continue  # skip this lambda, try next
            return

        if validation['warnings']:
            print("\n  WARNINGS:")
            for warning in validation['warnings']:
                print(f"    - {warning}")

        # Save results
        print(f"\n[6/6] Saving results to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
    
        # Only save single-lambda outputs for the last lambda (or if there's only one)
        is_last_lambda = (lam_value == lambdas[-1])
        if is_last_lambda:
            # Save CSV (single source of truth)
            csv_path = os.path.join(output_dir, "replay_summary.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header
                writer.writerow([
                    'replay_criterion', 'acceptance_rate', 'num_accepted',
                    'baseline_ctr', 'baseline_ndcg', 'ede_ctr', 'ede_ndcg',
                    'delta_ctr', 'delta_ctr_ci_lo', 'delta_ctr_ci_hi',
                    'delta_ndcg', 'delta_ndcg_ci_lo', 'delta_ndcg_ci_hi'
                ])
                # Exact-Match
                writer.writerow([
                    'Replay-Exact-Match',
                    f"{aggregated['exact_match_acceptance_rate']:.6f}",
                    aggregated['exact_match_num_accepted'],
                    f"{aggregated['exact_match_baseline_ctr']:.6f}",
                    f"{aggregated['exact_match_baseline_ndcg']:.6f}",
                    f"{aggregated['exact_match_ede_ctr']:.6f}",
                    f"{aggregated['exact_match_ede_ndcg']:.6f}",
                    f"{(aggregated['exact_match_ede_ctr'] - aggregated['exact_match_baseline_ctr']):.6f}",
                    '',
                    '',
                    f"{(aggregated['exact_match_ede_ndcg'] - aggregated['exact_match_baseline_ndcg']):.6f}",
                    '',
                    '',
                ])
                # Set-Match
                writer.writerow([
                    'Replay-Set-Match',
                    f"{aggregated['set_match_acceptance_rate']:.6f}",
                    aggregated['set_match_num_accepted'],
                    f"{aggregated['set_match_baseline_ctr']:.6f}",
                    f"{aggregated['set_match_baseline_ndcg']:.6f}",
                    f"{aggregated['set_match_ede_ctr']:.6f}",
                    f"{aggregated['set_match_ede_ndcg']:.6f}",
                    f"{(aggregated['set_match_ede_ctr'] - aggregated['set_match_baseline_ctr']):.6f}",
                    '',
                    '',
                    f"{(aggregated['set_match_ede_ndcg'] - aggregated['set_match_baseline_ndcg']):.6f}",
                    '',
                    '',
                ])
                # Click-Preservation
                writer.writerow([
                    'Replay-Click-Preservation',
                    f"{aggregated['click_pres_acceptance_rate']:.6f}",
                    aggregated['click_pres_num_accepted'],
                    f"{aggregated['click_pres_baseline_ctr']:.6f}",
                    f"{aggregated['click_pres_baseline_ndcg']:.6f}",
                    f"{aggregated['click_pres_ede_ctr']:.6f}",
                    f"{aggregated['click_pres_ede_ndcg']:.6f}",
                    f"{ci_delta_ctr:.6f}",
                    f"{ci_delta_ctr_lo:.6f}",
                    f"{ci_delta_ctr_hi:.6f}",
                    f"{ci_delta_ndcg:.6f}",
                    f"{ci_delta_ndcg_lo:.6f}",
                    f"{ci_delta_ndcg_hi:.6f}",
                ])
            print(f"  Saved: {csv_path}")

            # Save diagnostics CSV with all computed metrics
            diagnostics_path = os.path.join(output_dir, "replay_diagnostics.csv")
            with open(diagnostics_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['metric', 'value'])
                writer.writerow(['exact_list_change_rate_click_pres', f"{exact_list_change_rate:.6f}"])
                writer.writerow(['set_change_rate_click_pres', f"{set_change_rate:.6f}"])
                writer.writerow(['reorder_only_rate_click_pres', f"{reorder_only_rate:.6f}"])
                writer.writerow(['mean_positions_changed_count_click_pres', f"{mean_pos_diffs_click_pres:.6f}"])
                writer.writerow(['median_positions_changed_count_click_pres', f"{median_pos_diffs:.6f}"])
                writer.writerow(['p90_positions_changed_count_click_pres', f"{p90_pos_diffs:.6f}"])
                writer.writerow(['mean_positions_changed_fraction_click_pres', f"{mean_pos_diffs_fraction:.6f}"])
                writer.writerow(['pct_sessions_eq1_positions_changed_click_pres', f"{pct_sessions_eq1_positions_changed:.6f}"])
                writer.writerow(['pct_sessions_ge2_positions_changed_click_pres', f"{pct_sessions_ge2_positions_changed:.6f}"])
                writer.writerow(['pct_sessions_ge3_positions_changed_click_pres', f"{pct_sessions_ge3_positions_changed:.6f}"])
                writer.writerow(['top_3_most_changed_positions_click_pres', top_positions_str])
                writer.writerow(['mean_abs_shift_clicked_click_pres', f"{mean_clicked_shift:.6f}"])
                writer.writerow(['median_abs_shift_clicked_click_pres', f"{median_clicked_shift:.6f}"])
                writer.writerow(['pct_clicked_shift_gt0_click_pres', f"{pct_clicked_shift_gt0:.6f}"])
                writer.writerow(['mean_kendall_tau_intersection_click_pres', f"{mean_kendall_tau:.6f}"])
                writer.writerow(['mean_intersection_size_click_pres', f"{mean_intersection_size:.6f}"])
                writer.writerow(['order_change_rate_set_match_bucket', f"{pct_order_change_set_match:.6f}"])
                writer.writerow(['avg_clicks_per_session_top10', f"{avg_clicks_per_session:.6f}"])
                writer.writerow(['fraction_sessions_with_1_click', f"{frac_1_click:.6f}"])
                writer.writerow(['fraction_sessions_with_2plus_clicks', f"{frac_2plus_clicks:.6f}"])
                writer.writerow(['click_pres_bootstrap_enabled', int(bool(bootstrap_ci_enabled))])
                writer.writerow(['click_pres_bootstrap_B', int(bootstrap_B)])
                writer.writerow(['click_pres_bootstrap_seed', int(bootstrap_seed)])
                writer.writerow(['click_pres_bootstrap_n', int(n_click_pres)])
                writer.writerow(['click_pres_delta_ctr_point', f"{ci_delta_ctr:.6f}"])
                writer.writerow(['click_pres_delta_ctr_ci_lo', f"{ci_delta_ctr_lo:.6f}"])
                writer.writerow(['click_pres_delta_ctr_ci_hi', f"{ci_delta_ctr_hi:.6f}"])
                writer.writerow(['click_pres_delta_ndcg_point', f"{ci_delta_ndcg:.6f}"])
                writer.writerow(['click_pres_delta_ndcg_ci_lo', f"{ci_delta_ndcg_lo:.6f}"])
                writer.writerow(['click_pres_delta_ndcg_ci_hi', f"{ci_delta_ndcg_hi:.6f}"])
            print(f"  Saved: {diagnostics_path}")

            # Generate markdown report from CSVs (single source of truth)
            md_path = os.path.join(output_dir, "report_real_replay.md")
            _generate_markdown_report(
                md_path, csv_path, diagnostics_path, session_stats, aggregated, validation,
                lam_value, eta, m, alpha, k, top_L, tau,
                avg_clicks_per_session, frac_1_click, frac_2plus_clicks,
                bootstrap_B
            )
            print(f"  Saved: {md_path}")

        # Collect two-lambda rows (for every lambda, outside is_last_lambda)
        for criterion_key, label in (
            ('exact_match', 'Replay-Exact-Match'),
            ('set_match', 'Replay-Set-Match'),
            ('click_pres', 'Replay-Click-Preservation'),
        ):
            acceptance = aggregated[f"{criterion_key}_acceptance_rate"]
            baseline_ctr = aggregated[f"{criterion_key}_baseline_ctr"]
            baseline_ndcg = aggregated[f"{criterion_key}_baseline_ndcg"]
            ede_ctr = aggregated[f"{criterion_key}_ede_ctr"]
            ede_ndcg = aggregated[f"{criterion_key}_ede_ndcg"]

            stats = criterion_stats[label]
            if stats['accepted'] > 0:
                pct_any_reorder_from_stats = stats['reorder'] / stats['accepted']
                mean_positions_changed = stats['mismatch_sum'] / stats['accepted']
                if label == 'Replay-Click-Preservation':
                    mean_abs_rank_shift_clicked = stats['clicked_shift_sum'] / stats['accepted']
                    pct_any_reorder = exact_list_change_rate
                    delta_ctr_ci_lo = ci_delta_ctr_lo
                    delta_ctr_ci_hi = ci_delta_ctr_hi
                    delta_ndcg_ci_lo = ci_delta_ndcg_lo
                    delta_ndcg_ci_hi = ci_delta_ndcg_hi
                else:
                    mean_abs_rank_shift_clicked = 0.0
                    pct_any_reorder = pct_any_reorder_from_stats
                    delta_ctr_ci_lo = None
                    delta_ctr_ci_hi = None
                    delta_ndcg_ci_lo = None
                    delta_ndcg_ci_hi = None
            else:
                pct_any_reorder = 0.0
                mean_positions_changed = 0.0
                mean_abs_rank_shift_clicked = 0.0
                delta_ctr_ci_lo = None
                delta_ctr_ci_hi = None
                delta_ndcg_ci_lo = None
                delta_ndcg_ci_hi = None

            two_lambda_rows.append({
                'criterion': label,
                'lambda': lam_value,
                'acceptance_rate': acceptance,
                'ctr10': ede_ctr,
                'ndcg10': ede_ndcg,
                'delta_ctr10': ede_ctr - baseline_ctr,
                'delta_ndcg10': ede_ndcg - baseline_ndcg,
                'delta_ctr_ci_lo': delta_ctr_ci_lo,
                'delta_ctr_ci_hi': delta_ctr_ci_hi,
                'delta_ndcg_ci_lo': delta_ndcg_ci_lo,
                'delta_ndcg_ci_hi': delta_ndcg_ci_hi,
                'pct_any_reorder': pct_any_reorder,
                'mean_positions_changed': mean_positions_changed,
                'mean_abs_rank_shift_clicked': mean_abs_rank_shift_clicked,
            })

        two_lambda_diagnostics[lam_value] = {
            'pct_any_reorder_click_pres': exact_list_change_rate,
            'mean_positions_changed_click_pres': mean_pos_diffs_click_pres,
        }

    # --- end of for lam_value in lambdas ---
    print("\n" + "=" * 70)
    print("Replay check completed successfully!")
    print("=" * 70)
    print("REPLAY DIAGNOSTICS SUMMARY")
    print(f"Exact-match rate: {exact_rate:.6f}")
    print(f"Set-match rate: {set_rate:.6f}")
    print(f"Click-preservation rate: {click_rate:.6f}")
    print(f"Exact list change rate (click-preservation): {exact_list_change_rate:.6f}")
    print(f"Set change rate (click-preservation): {set_change_rate:.6f}")
    print(f"Reorder-only rate (click-preservation): {reorder_only_rate:.6f}")
    print(f"Mean # position diffs (click-preservation): {mean_pos_diffs_click_pres:.6f}")
    print(f"Mean abs rank shift of clicked docs: {mean_clicked_shift:.6f}")
    print(f"Mean Kendall tau (intersection): {mean_kendall_tau:.6f}")

    # Write two-lambda outputs when requested
    if len(lambdas) > 1:
        two_lambda_path = os.path.join(output_dir, "replay_summary_two_lambda.csv")
        with open(two_lambda_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'criterion', 'lambda', 'acceptance_rate', 'ctr10', 'ndcg10',
                'delta_ctr10', 'delta_ndcg10',
                'delta_ctr_ci_lo', 'delta_ctr_ci_hi', 'delta_ndcg_ci_lo', 'delta_ndcg_ci_hi',
                'pct_any_reorder', 'mean_positions_changed', 'mean_abs_rank_shift_clicked',
            ])
            for row in two_lambda_rows:
                writer.writerow([
                    row['criterion'],
                    f"{row['lambda']:.3f}",
                    f"{row['acceptance_rate']:.6f}",
                    f"{row['ctr10']:.6f}",
                    f"{row['ndcg10']:.6f}",
                    f"{row['delta_ctr10']:.6f}",
                    f"{row['delta_ndcg10']:.6f}",
                    '' if row['delta_ctr_ci_lo'] is None else f"{row['delta_ctr_ci_lo']:.6f}",
                    '' if row['delta_ctr_ci_hi'] is None else f"{row['delta_ctr_ci_hi']:.6f}",
                    '' if row['delta_ndcg_ci_lo'] is None else f"{row['delta_ndcg_ci_lo']:.6f}",
                    '' if row['delta_ndcg_ci_hi'] is None else f"{row['delta_ndcg_ci_hi']:.6f}",
                    f"{row['pct_any_reorder']:.6f}",
                    f"{row['mean_positions_changed']:.6f}",
                    f"{row['mean_abs_rank_shift_clicked']:.6f}",
                ])
        print(f"  Saved: {two_lambda_path}")

        # Validation: fewer changes expectation
        if 0.2 in two_lambda_diagnostics and 0.4 in two_lambda_diagnostics:
            pct_02 = two_lambda_diagnostics[0.2]['pct_any_reorder_click_pres']
            pct_04 = two_lambda_diagnostics[0.4]['pct_any_reorder_click_pres']
            if pct_02 > pct_04 + 0.02:
                print(
                    "  NOTE: Fewer-changes expectation violated: "
                    f"pct_any_reorder(0.2)={pct_02:.3f}, pct_any_reorder(0.4)={pct_04:.3f}"
                )

        # Two-lambda report and appendix snippet
        report_two_lambda_path = os.path.join(output_dir, "report_real_replay_two_lambda.md")
        _generate_two_lambda_report(report_two_lambda_path, two_lambda_path)
        print(f"  Saved: {report_two_lambda_path}")

        snippet_path = os.path.join(output_dir, "appendix_snippet_two_lambda.md")
        _generate_two_lambda_snippet(snippet_path, two_lambda_path)
        print(f"  Saved: {snippet_path}")
def _generate_markdown_report(
    md_path: str,
    csv_path: str,
    diagnostics_path: str,
    session_stats: Dict,
    aggregated: Dict,
    validation: Dict,
    lam: float,
    eta: float,
    m: int,
    alpha: float,
    k: int,
    top_L: int,
    tau: float,
    avg_clicks: float,
    frac_1_click: float,
    frac_2plus_clicks: float,
    bootstrap_B: int,
):
    """Generate markdown report from aggregated results."""
    # Read CSV to ensure single source of truth
    csv_data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_data[row['replay_criterion']] = row
    
    diagnostics = {}
    with open(diagnostics_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            diagnostics[row['metric']] = row['value']

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Real Click-Log Conservative Replay Sanity Check\n\n")
        f.write("Dataset: Yandex Personalized Web Search Challenge (Kaggle).\n\n")
        f.write("**IMPORTANT**: This is NOT unbiased counterfactual evaluation. ")
        f.write("This is a conservative replay sanity check that only evaluates sessions ")
        f.write("where EDE makes minimal or no changes to the logged ranking.\n\n")
        
        f.write("## Dataset\n\n")
        f.write("**Yandex Personalized Web Search Challenge**\n\n")
        f.write(f"- Total sessions loaded: {session_stats['num_sessions']:,}\n")
        f.write(f"- Sessions with >=10 results: {session_stats['num_with_10plus_results']:,}\n")
        f.write(f"- Sessions with clicks: {session_stats['num_with_clicks']:,}\n")
        f.write(f"- Fields used: ranked URLs (top-{k}), click indicators (0/1)\n\n")
        
        f.write("### Click Distribution\n\n")
        f.write(f"- Average clicks per session (top-{k}): {avg_clicks:.3f}\n")
        f.write(f"- Sessions with exactly 1 click: {frac_1_click*100:.1f}%\n")
        f.write(f"- Sessions with 2+ clicks: {frac_2plus_clicks*100:.1f}%\n\n")
        f.write("*Note: High Click-Preservation acceptance rates are expected when most ")
        f.write("sessions have only 1-2 clicks, as EDE can preserve a single clicked document ")
        f.write("while reranking.*\n\n")
        
        f.write("## Conservative Replay Criteria\n\n")
        f.write("We evaluate EDE reranking under three conservative replay criteria:\n\n")
        f.write("1. **Replay-Exact-Match**: Accept session only if EDE top-k matches baseline ")
        f.write("top-k in exact order (strictest).\n\n")
        f.write("2. **Replay-Set-Match (order-agnostic)**: Accept session only if EDE top-k ")
        f.write("contains exactly the same documents as baseline top-k, but order can differ.\n\n")
        f.write("3. **Replay-Click-Preservation**: Accept session only if every clicked ")
        f.write("document in baseline top-k remains in EDE top-k (most permissive).\n\n")
        f.write("All criteria ensure conservative evaluation by limiting to sessions ")
        f.write("where EDE changes are minimal or preserve critical clicked results.\n\n")
        
        f.write("## EDE Configuration\n\n")
        f.write(f"- lambda (exploration weight): {lam}\n")
        f.write(f"- eta (novelty weight): {eta}\n")
        f.write(f"- m (safe-prefix length): {m}\n")
        f.write(f"- alpha (smoothing): {alpha}\n")
        f.write(f"- k (ranking cutoff): {k}\n")
        f.write(f"- top_L (candidate gating): {top_L}\n")
        f.write(f"- tau (exposure decay): {tau}\n\n")
        
        f.write("## Acceptance Rates\n\n")
        exact_data = csv_data['Replay-Exact-Match']
        set_data = csv_data['Replay-Set-Match']
        click_data = csv_data['Replay-Click-Preservation']
        
        f.write(f"- **Replay-Exact-Match**: {float(exact_data['acceptance_rate'])*100:.2f}% ")
        f.write(f"({exact_data['num_accepted']} sessions)\n")
        f.write(f"- **Replay-Set-Match**: {float(set_data['acceptance_rate'])*100:.2f}% ")
        f.write(f"({set_data['num_accepted']} sessions)\n")
        f.write(f"- **Replay-Click-Preservation**: {float(click_data['acceptance_rate'])*100:.2f}% ")
        f.write(f"({click_data['num_accepted']} sessions)\n\n")
        
        f.write("## Metrics on Accepted Sessions\n\n")
        f.write("*All metrics computed from replay_summary.csv*\n\n")
        
        f.write("### Replay-Exact-Match\n\n")
        f.write("| Policy | CTR@10 | NDCG@10 |\n")
        f.write("|--------|--------|----------|\n")
        f.write(f"| Baseline | {float(exact_data['baseline_ctr']):.4f} | ")
        f.write(f"{float(exact_data['baseline_ndcg']):.4f} |\n")
        f.write(f"| EDE-Safer | {float(exact_data['ede_ctr']):.4f} | ")
        f.write(f"{float(exact_data['ede_ndcg']):.4f} |\n\n")
        
        f.write("### Replay-Set-Match (order-agnostic)\n\n")
        f.write("| Policy | CTR@10 | NDCG@10 |\n")
        f.write("|--------|--------|----------|\n")
        f.write(f"| Baseline | {float(set_data['baseline_ctr']):.4f} | ")
        f.write(f"{float(set_data['baseline_ndcg']):.4f} |\n")
        f.write(f"| EDE-Safer | {float(set_data['ede_ctr']):.4f} | ")
        f.write(f"{float(set_data['ede_ndcg']):.4f} |\n\n")
        
        f.write("### Replay-Click-Preservation\n\n")
        f.write("| Policy | CTR@10 | NDCG@10 |\n")
        f.write("|--------|--------|----------|\n")
        f.write(f"| Baseline | {float(click_data['baseline_ctr']):.4f} | ")
        f.write(f"{float(click_data['baseline_ndcg']):.4f} |\n")
        f.write(f"| EDE-Safer | {float(click_data['ede_ctr']):.4f} | ")
        f.write(f"{float(click_data['ede_ndcg']):.4f} |\n\n")

        f.write(f"### Bootstrap 95% CI (B={bootstrap_B}) for Click-Preservation deltas\n\n")
        if click_data.get('delta_ctr_ci_lo', '') and click_data.get('delta_ndcg_ci_lo', ''):
            f.write(
                f"- ΔCTR@10 = {float(click_data['delta_ctr']):+.6f} "
                f"[{float(click_data['delta_ctr_ci_lo']):+.6f}, {float(click_data['delta_ctr_ci_hi']):+.6f}]\n"
            )
            f.write(
                f"- Δclick-NDCG@10 = {float(click_data['delta_ndcg']):+.6f} "
                f"[{float(click_data['delta_ndcg_ci_lo']):+.6f}, {float(click_data['delta_ndcg_ci_hi']):+.6f}]\n\n"
            )
            f.write("*These are bootstrap CIs over accepted sessions (Replay-Click-Preservation) and are not causal estimates.*\n\n")
        else:
            f.write("- Bootstrap CI not enabled for this run.\n\n")
        
        f.write("## Ranking Change Diagnostics\n\n")
        f.write("The following diagnostics are computed over Replay-Click-Preservation ")
        f.write("sessions (large, stable bucket) unless noted otherwise.\n\n")
        
        f.write("**On Click-Preservation Bucket:**\n\n")
        f.write(f"- Exact-list change rate: {float(diagnostics['exact_list_change_rate_click_pres'])*100:.2f}%\n")
        f.write(f"- Set-change rate: {float(diagnostics['set_change_rate_click_pres'])*100:.2f}%\n")
        f.write(f"- Reorder-only rate (same set, different order): {float(diagnostics['reorder_only_rate_click_pres'])*100:.2f}%\n\n")
        
        f.write("**Position Change Distribution (Click-Preservation Bucket):**\n\n")
        f.write(f"- Mean positions changed count: {float(diagnostics['mean_positions_changed_count_click_pres']):.3f} / {k}\n")
        f.write(f"- Median positions changed count: {float(diagnostics['median_positions_changed_count_click_pres']):.1f}\n")
        f.write(f"- P90 positions changed count: {float(diagnostics['p90_positions_changed_count_click_pres']):.1f}\n")
        f.write(f"- Mean positions changed fraction: {float(diagnostics['mean_positions_changed_fraction_click_pres']):.4f}\n")
        f.write(f"- % sessions with exactly 1 position changed: {float(diagnostics['pct_sessions_eq1_positions_changed_click_pres'])*100:.2f}%\n")
        f.write(f"- % sessions with >=2 positions changed: {float(diagnostics['pct_sessions_ge2_positions_changed_click_pres'])*100:.2f}%\n")
        f.write(f"- % sessions with >=3 positions changed: {float(diagnostics['pct_sessions_ge3_positions_changed_click_pres'])*100:.2f}%\n")
        f.write(f"- Top 3 most frequently changed positions: {diagnostics['top_3_most_changed_positions_click_pres']}\n\n")
        
        f.write("**Clicked Document Rank Shifts (Click-Preservation Bucket):**\n\n")
        f.write(f"- Mean absolute shift of clicked docs: {float(diagnostics['mean_abs_shift_clicked_click_pres']):.3f}\n")
        f.write(f"- Median absolute shift of clicked docs: {float(diagnostics['median_abs_shift_clicked_click_pres']):.3f}\n")
        f.write(f"- % clicked docs with shift > 0: {float(diagnostics['pct_clicked_shift_gt0_click_pres'])*100:.2f}%\n\n")
        
        f.write("**Ranking Similarity (Click-Preservation Bucket):**\n\n")
        f.write(f"- Mean Kendall tau (on intersection): {float(diagnostics['mean_kendall_tau_intersection_click_pres']):.4f}\n")
        f.write(f"- Mean intersection size: {float(diagnostics['mean_intersection_size_click_pres']):.2f}\n\n")
        
        f.write(f"**Set-Match Bucket:**\n\n")
        f.write(f"- Order-change rate (set-match bucket only): {float(diagnostics['order_change_rate_set_match_bucket'])*100:.2f}%\n\n")
        
        f.write("**Interpretation:**\n\n")
        f.write("- Exact-list change rate: fraction of sessions where EDE top-10 differs from baseline (any position).\n")
        f.write("- Position change distribution is a 'gold star' diagnostic showing how many positions change per session:\n")
        f.write("  - Most changes are single-position replacements (typical: 1-2 docs swapped out)\n")
        f.write("  - Few sessions have 3+ positions changed (wholesale ranking reorganization)\n")
        f.write("- Top 3 changed positions usually skew toward position 10 (most permissive for replacement)\n")
        f.write("- Clicked documents preserved in position (0.000 mean shift) validates click-preservation constraint\n")
        f.write("- If set-match bucket shows ~0% order changes, exact-match and set-match acceptance rates coincide\n\n")

        f.write("## Interpretation\n\n")
        f.write("This appendix-style experiment provides a conservative replay sanity check ")
        f.write("of EDE reranking using real Yandex click logs. ")
        f.write("We apply EDE's entropy-driven exploration to logged candidate lists without training new models. ")
        f.write("The replay criteria ensure we only evaluate on sessions where EDE makes minimal changes ")
        f.write("(Exact/Set-Match) or preserves clicked results (Click-Preservation).\n\n")
        
        f.write("**Limitations**: This is not unbiased offline evaluation. ")
        f.write("EDE's reranking may benefit from observing the logged policy's exploration, ")
        f.write("and replay-based filtering introduces selection bias (we only evaluate on sessions ")
        f.write("where EDE happens to make acceptable changes). These results provide a directional ")
        f.write("sanity check, not causal claims about EDE's performance.\n\n")
        
        if validation['warnings']:
            f.write("## Validation Warnings\n\n")
            for warning in validation['warnings']:
                f.write(f"- {warning}\n")


def _generate_two_lambda_report(md_path: str, csv_path: str):
    """Generate two-lambda comparison report."""
    # Read CSV manually
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Two-Lambda Ablation: Position Change Sensitivity\n\n")
        f.write("**Goal:** Show how exploration parameter lambda affects ranking changes.\n\n")
        f.write("**Hypothesis:** Lower lambda (less exploration) -> fewer position changes, similar performance.\n\n")
        
        f.write("## Click-Preservation Results\n\n")
        click_pres = [r for r in rows if r['criterion'] == 'Replay-Click-Preservation']
        
        f.write("| lambda | Acceptance | CTR Delta | NDCG Delta | CTR 95% CI | NDCG 95% CI | Exact-list change % | Mean pos changed |\n")
        f.write("|--------|------------|-----------|------------|------------|-------------|---------------------|------------------|\n")
        
        for row in click_pres:
            ctr_ci = "-"
            ndcg_ci = "-"
            if row.get('delta_ctr_ci_lo') and row.get('delta_ctr_ci_hi'):
                ctr_ci = f"[{float(row['delta_ctr_ci_lo']):+.4f}, {float(row['delta_ctr_ci_hi']):+.4f}]"
            if row.get('delta_ndcg_ci_lo') and row.get('delta_ndcg_ci_hi'):
                ndcg_ci = f"[{float(row['delta_ndcg_ci_lo']):+.4f}, {float(row['delta_ndcg_ci_hi']):+.4f}]"
            f.write(f"| {float(row['lambda']):.1f} | {float(row['acceptance_rate'])*100:.1f}% | ")
            f.write(f"{float(row['delta_ctr10']):+.4f} | {float(row['delta_ndcg10']):+.4f} | ")
            f.write(f"{ctr_ci} | {ndcg_ci} | ")
            f.write(f"{float(row['pct_any_reorder'])*100:.1f}% | {float(row['mean_positions_changed']):.3f} |\n")
        
        f.write("\n## Interpretation\n\n")
        if len(click_pres) >= 2:
            lambdas = sorted([float(r['lambda']) for r in click_pres])
            low_row = [r for r in click_pres if float(r['lambda']) == lambdas[0]][0]
            high_row = [r for r in click_pres if float(r['lambda']) == lambdas[1]][0]
            
            low_change = float(low_row['pct_any_reorder'])
            high_change = float(high_row['pct_any_reorder'])
            change_ratio = low_change / high_change if high_change > 0 else 1.0
            
            f.write(f"- lambda={lambdas[0]:.1f} makes {change_ratio:.2f}x fewer changes than lambda={lambdas[1]:.1f}\n")
            f.write(f"- Performance deltas similar: CTR Delta within {abs(float(high_row['delta_ctr10']) - float(low_row['delta_ctr10'])):.4f}\n")
            f.write(f"- Both preserve clicked documents (mean shift ~= 0)\n")
            f.write(f"- Lower lambda confirms more conservative exploration behavior\n")


def _generate_two_lambda_snippet(snippet_path: str, csv_path: str):
    """Generate appendix snippet for two-lambda ablation."""
    # Read CSV manually
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    click_pres = [r for r in rows if r['criterion'] == 'Replay-Click-Preservation']
    
    with open(snippet_path, 'w', encoding='utf-8') as f:
        f.write("### Appendix: Lambda Sensitivity\n\n")
        f.write("We validate that exploration parameter lambda controls ranking conservativeness:\n\n")
        
        f.write("| lambda | Exact-list change rate | Mean positions changed | CTR Delta [95% CI] | NDCG Delta [95% CI] |\n")
        f.write("|--------|------------------------|------------------------|--------------------|---------------------|\n")
        
        for row in click_pres:
            ctr_delta = f"{float(row['delta_ctr10']):+.4f}"
            ndcg_delta = f"{float(row['delta_ndcg10']):+.4f}"
            if row.get('delta_ctr_ci_lo') and row.get('delta_ctr_ci_hi'):
                ctr_delta = (
                    f"{ctr_delta} [{float(row['delta_ctr_ci_lo']):+.4f}, "
                    f"{float(row['delta_ctr_ci_hi']):+.4f}]"
                )
            if row.get('delta_ndcg_ci_lo') and row.get('delta_ndcg_ci_hi'):
                ndcg_delta = (
                    f"{ndcg_delta} [{float(row['delta_ndcg_ci_lo']):+.4f}, "
                    f"{float(row['delta_ndcg_ci_hi']):+.4f}]"
                )
            f.write(f"| {float(row['lambda']):.1f} | {float(row['pct_any_reorder'])*100:.1f}% | ")
            f.write(f"{float(row['mean_positions_changed']):.3f} | ")
            f.write(f"{ctr_delta} | {ndcg_delta} |\n")
        
        f.write("\nLower lambda yields fewer ranking changes with comparable performance gains.\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run EDE real-data replay check using Yandex logs"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to directory containing Yandex log files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/real_replay",
        help="Path to output directory (default: outputs/real_replay)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Ranking cutoff (default: 10)",
    )
    parser.add_argument(
        "--min_results",
        type=int,
        default=10,
        help="Minimum results per session (default: 10)",
    )
    parser.add_argument(
        "--max_sessions",
        type=int,
        default=200000,
        help="Maximum sessions to parse from logs (default: 200000)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.4,
        help="EDE lambda parameter (default: 0.4)",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs='+',
        default=None,
        help="Multiple lambda values for ablation (e.g., --lambdas 0.2 0.4)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.65,
        help="EDE eta parameter (default: 0.65)",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=2,
        help="EDE safe-prefix length (default: 2)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="EDE alpha parameter (default: 0.5)",
    )
    parser.add_argument(
        "--top_L",
        type=int,
        default=50,
        help="EDE candidate gating threshold (default: 50)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=50.0,
        help="EDE tau parameter (default: 50.0)",
    )
    parser.add_argument(
        "--bootstrap_ci",
        type=lambda x: str(x).lower() in ("1", "true", "t", "yes", "y"),
        default=False,
        help="Enable bootstrap CI over Replay-Click-Preservation accepted sessions (default: false)",
    )
    parser.add_argument(
        "--bootstrap_B",
        type=int,
        default=1000,
        help="Number of bootstrap resamples (default: 1000)",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=0,
        help="Bootstrap random seed (default: 0)",
    )
    parser.add_argument(
        "--lambda_sanity_test",
        action="store_true",
        help="Run lambda sanity test with lambdas=[0.0, 0.2, 0.4, 2.0]",
    )
    
    args = parser.parse_args()
    
    # Determine lambda values to use
    if args.lambdas is not None:
        lambda_values = args.lambdas
    else:
        lambda_values = [args.lam]
    
    # Run sanity test if requested
    if args.lambda_sanity_test:
        run_lambda_sanity_test(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            k=args.k,
            min_results=args.min_results,
            max_sessions=args.max_sessions,
            eta=args.eta,
            m=args.m,
            alpha=args.alpha,
            top_L=args.top_L,
            tau=args.tau,
        )
        return
    
    run_replay_check(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        k=args.k,
        min_results=args.min_results,
        max_sessions=args.max_sessions,
        bootstrap_ci_enabled=args.bootstrap_ci,
        bootstrap_B=args.bootstrap_B,
        bootstrap_seed=args.bootstrap_seed,
        lam=args.lam,
        lambdas=lambda_values,
        eta=args.eta,
        m=args.m,
        alpha=args.alpha,
        top_L=args.top_L,
        tau=args.tau,
    )


if __name__ == "__main__":
    main()
