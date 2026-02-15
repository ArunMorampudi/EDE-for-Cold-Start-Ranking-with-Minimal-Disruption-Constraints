"""Load and parse Yandex Personalized Web Search Challenge logs."""
import os
from typing import List, Dict, Iterable
import numpy as np


class YandexSession:
    """Container for a single search session."""
    
    def __init__(
        self,
        session_id: int,
        query_id: int,
        timestamp: int,
        ranked_docs: List[int],
        clicks: List[int],
    ):
        """
        Initialize session.
        
        Args:
            session_id: unique session identifier
            query_id: query identifier  
            timestamp: timestamp/order of session
            ranked_docs: list of doc/url ids in shown order
            clicks: 0/1 click indicators aligned to ranked_docs
        """
        self.session_id = session_id
        self.query_id = query_id
        self.timestamp = timestamp
        self.ranked_docs = ranked_docs
        self.clicks = clicks
        
    def has_click(self) -> bool:
        """Check if session has at least one click."""
        return sum(self.clicks) > 0
    
    def num_results(self) -> int:
        """Return number of results shown."""
        return len(self.ranked_docs)
    
    def validate(self) -> bool:
        """Validate session structure."""
        return len(self.ranked_docs) == len(self.clicks)


def load_yandex_sessions(
    data_dir: str,
    k_min: int = 10,
    max_sessions: int = 200_000,
) -> List[YandexSession]:
    """
    Load Yandex Personalized Web Search Challenge logs into sessions.

    Args:
        data_dir: directory containing Kaggle Yandex log files
        k_min: minimum number of displayed results required per session
        max_sessions: maximum number of raw sessions to parse

    Returns:
        list of YandexSession objects sorted by timestamp
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data not found at {data_dir}. "
            "Please download the Kaggle Yandex dataset and pass --data_dir."
        )

    sessions = []

    # Scan for log files
    log_files: List[str] = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith('.txt') or fname.endswith('.tsv'):
                log_files.append(os.path.join(root, fname))

    if not log_files:
        raise FileNotFoundError(
            f"No log files found in {data_dir}. "
            "Please download the Kaggle Yandex dataset and pass --data_dir."
        )

    log_files = sorted(log_files)
    print(f"Found {len(log_files)} log file(s) to process")

    # Sessions can span across multiple files, so we must process sequentially
    # (parallel processing would break session continuity)
    print("Processing files sequentially to maintain session continuity...")
    
    session_data: Dict[int, Dict[str, object]] = {}
    stop_requested = False
    for log_file in log_files:
        file_size = os.path.getsize(log_file)
        file_size_mb = file_size / (1024 * 1024)
        print(f"Processing {os.path.basename(log_file)} ({file_size_mb:.1f} MB)...")
        
        progress_interval = 1000000  # Report every 1 million lines
        with open(log_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num == 1 and line.startswith('SessionId'):
                    continue
                
                # Progress reporting for large files
                if line_num % progress_interval == 0:
                    print(f"  [{os.path.basename(log_file)}] Processed {line_num:,} lines ({len(session_data):,} sessions so far)...")

                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue

                try:
                    session_id = int(parts[0])
                    
                    # Yandex format:
                    # M: SessionID M Day UserID
                    # Q/T: SessionID TimePassed Q|T SERPID QueryID Terms URL1,Domain1 ...
                    # C: SessionID TimePassed C SERPID URLID
                    if len(parts) > 1 and parts[1] == 'M':
                        type_code = 'M'
                    elif len(parts) > 2 and parts[2] in {'Q', 'C', 'T'}:
                        type_code = parts[2]
                    else:
                        continue

                    if (
                        max_sessions is not None
                        and session_id not in session_data
                        and len(session_data) >= max_sessions
                    ):
                        stop_requested = True
                        break
                    
                    if type_code == 'M':
                        # Session metadata - initialize session if needed
                        if session_id not in session_data:
                            session_data[session_id] = {
                                'timestamp': session_id,  # Use session_id as timestamp
                                'query_id': -1,
                                'ranked_docs': [],
                                'clicks': [],
                                'doc_positions': {},
                            }
                        continue
                    
                    elif type_code == 'Q':
                        # Query with URLs shown
                        # Format: SessionID TimePassed Q SERPID QueryID Terms URL1,Domain1 ...
                        if len(parts) < 7:
                            continue
                        
                        if session_id not in session_data:
                            session_data[session_id] = {
                                'timestamp': session_id,
                                'query_id': -1,
                                'ranked_docs': [],
                                'clicks': [],
                                'doc_positions': {},
                            }
                        
                        session = session_data[session_id]
                        query_id = int(parts[4]) if parts[4].isdigit() else -1
                        
                        if session['query_id'] == -1:
                            session['query_id'] = query_id
                        
                        # URLs are in positions 6+, format: url_id,domain_id
                        for i in range(6, len(parts)):
                            if not parts[i]:
                                continue
                            url_info = parts[i].split(',')
                            if not url_info[0]:
                                continue
                            try:
                                url_id = int(url_info[0])
                                if url_id not in session['doc_positions']:
                                    pos = len(session['ranked_docs'])
                                    session['ranked_docs'].append(url_id)
                                    session['clicks'].append(0)
                                    session['doc_positions'][url_id] = pos
                            except (ValueError, IndexError):
                                continue
                    
                    elif type_code == 'C':
                        # Click event
                        # Format: SessionID TimePassed C SERPID URLID
                        if len(parts) < 5:
                            continue
                        
                        if session_id not in session_data:
                            continue  # Skip clicks for unknown sessions
                        
                        session = session_data[session_id]
                        url_id = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else -1
                        
                        if url_id != -1 and url_id in session['doc_positions']:
                            pos = session['doc_positions'][url_id]
                            session['clicks'][pos] = 1
                    
                    elif type_code == 'T':
                        # Test query/reformulation line (same layout as Q)
                        if len(parts) < 7:
                            continue
                        
                        if session_id not in session_data:
                            session_data[session_id] = {
                                'timestamp': session_id,
                                'query_id': -1,
                                'ranked_docs': [],
                                'clicks': [],
                                'doc_positions': {},
                            }
                        
                        session = session_data[session_id]
                        
                        query_id = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else -1
                        if query_id != -1:
                            session['query_id'] = query_id
                        
                        # URLs are in positions 6+
                        for i in range(6, len(parts)):
                            if not parts[i]:
                                continue
                            url_info = parts[i].split(',')
                            if not url_info[0]:
                                continue
                            try:
                                url_id = int(url_info[0])
                                if url_id not in session['doc_positions']:
                                    pos = len(session['ranked_docs'])
                                    session['ranked_docs'].append(url_id)
                                    session['clicks'].append(0)
                                    session['doc_positions'][url_id] = pos
                            except (ValueError, IndexError):
                                continue
                    
                except (ValueError, IndexError):
                    continue

            if stop_requested:
                print(f"  Reached max_sessions={max_sessions:,}; stopping parse early")
                break
        
        # File processing complete
        print(f"  Completed {os.path.basename(log_file)}: {line_num:,} lines processed, {len(session_data):,} sessions total")

        if stop_requested:
            break

    total_sessions = len(session_data)
    alignment_errors = 0
    sessions_with_10plus = 0
    sessions_with_clicks = 0
    click_counts = []
    
    print(f"  Total raw sessions from files: {total_sessions:,}")

    # CTR-by-position stats for sessions with >= k_min results
    max_pos = 10
    pos_clicks = np.zeros(max_pos, dtype=np.float64)
    pos_impressions = np.zeros(max_pos, dtype=np.float64)

    for session_id, data in session_data.items():
        ranked_docs = data['ranked_docs']
        clicks = data['clicks']
        if len(ranked_docs) != len(clicks):
            alignment_errors += 1
            continue

        if len(ranked_docs) >= k_min:
            sessions_with_10plus += 1
            for i in range(min(max_pos, len(ranked_docs))):
                pos_impressions[i] += 1
                if clicks[i] == 1:
                    pos_clicks[i] += 1

        if len(ranked_docs) < k_min:
            continue
        if sum(clicks) == 0:
            continue

        click_count = int(sum(clicks))
        click_counts.append(click_count)
        if click_count > 0:
            sessions_with_clicks += 1

        sess = YandexSession(
            session_id=session_id,
            query_id=data['query_id'],
            timestamp=data['timestamp'],
            ranked_docs=ranked_docs,
            clicks=clicks,
        )

        if sess.validate():
            sessions.append(sess)
    
    print(f"  Sessions with >=10 results: {sessions_with_10plus:,}")
    print(f"  Sessions with >=10 results AND >=1 click: {len(sessions):,}")

    if total_sessions == 0:
        raise RuntimeError("No sessions found in Yandex logs.")

    alignment_rate = 1.0 - (alignment_errors / total_sessions)
    if alignment_rate < 0.99:
        raise RuntimeError(
            f"Click alignment failed: {alignment_rate*100:.2f}% aligned "
            "(expected >99%)."
        )

    # Sort by timestamp for streaming counters
    sessions.sort(key=lambda s: (s.timestamp, s.session_id))

    print(f"Total sessions parsed: {total_sessions}")
    if total_sessions > 0:
        pct_10plus = (sessions_with_10plus / total_sessions) * 100
        print(f"% sessions with >=10 results: {pct_10plus:.1f}%")

    if sessions:
        avg_clicks = float(np.mean(click_counts)) if click_counts else 0.0
        frac_1_click = sum(1 for c in click_counts if c == 1) / len(click_counts) if click_counts else 0.0
        frac_2plus = sum(1 for c in click_counts if c >= 2) / len(click_counts) if click_counts else 0.0
        print(f"Avg clicks/session (usable): {avg_clicks:.3f}")
        print(f"Sessions with 1 click (usable): {frac_1_click*100:.1f}%")
        print(f"Sessions with 2+ clicks (usable): {frac_2plus*100:.1f}%")

    ctr_positions = {}
    for pos in [1, 5, 10]:
        idx = pos - 1
        if idx < max_pos and pos_impressions[idx] > 0:
            ctr_positions[pos] = pos_clicks[idx] / pos_impressions[idx]

    if ctr_positions:
        print("CTR by position (1,5,10): " + ", ".join(
            f"{pos}:{ctr_positions[pos]:.3f}" for pos in sorted(ctr_positions)
        ))
        if not (ctr_positions.get(1, 1.0) >= ctr_positions.get(5, 1.0) >= ctr_positions.get(10, 1.0)):
            print("WARNING: CTR by position not decreasing (1 >= 5 >= 10 expected).")

    print(f"Loaded {len(sessions)} usable sessions (>={k_min} results, >=1 click)")
    return sessions


def parse_yandex_logs(data_dir: str, min_results: int = 10) -> List[YandexSession]:
    """Backward-compatible wrapper for legacy usage."""
    return load_yandex_sessions(data_dir=data_dir, k_min=min_results)


def validate_loaded_sessions(sessions: List[YandexSession]) -> Dict[str, any]:
    """
    Validate loaded sessions and return statistics.
    
    Args:
        sessions: list of loaded sessions
        
    Returns:
        dict with validation statistics
    """
    if not sessions:
        return {
            'num_sessions': 0,
            'num_with_10plus_results': 0,
            'num_with_clicks': 0,
            'alignment_errors': 0,
            'validation_passed': False,
            'warnings': ['No sessions loaded'],
        }
    
    stats = {
        'num_sessions': len(sessions),
        'num_with_10plus_results': 0,
        'num_with_clicks': 0,
        'alignment_errors': 0,
        'validation_passed': True,
        'warnings': [],
    }
    
    for sess in sessions:
        if sess.num_results() >= 10:
            stats['num_with_10plus_results'] += 1
        if sess.has_click():
            stats['num_with_clicks'] += 1
        if not sess.validate():
            stats['alignment_errors'] += 1
    
    # Check validation criteria
    if stats['num_sessions'] < 50000:
        stats['warnings'].append(
            f"Only {stats['num_sessions']} sessions loaded (expected >=50k)"
        )
    
    pct_with_10plus = (stats['num_with_10plus_results'] / stats['num_sessions']) * 100
    if pct_with_10plus < 70:
        stats['warnings'].append(
            f"Only {pct_with_10plus:.1f}% sessions have >=10 results (expected >=70%)"
        )
    
    pct_alignment = ((stats['num_sessions'] - stats['alignment_errors']) / stats['num_sessions']) * 100
    if pct_alignment < 99:
        stats['validation_passed'] = False
        stats['warnings'].append(
            f"Only {pct_alignment:.1f}% sessions have aligned clicks (expected >99%)"
        )
    
    return stats
