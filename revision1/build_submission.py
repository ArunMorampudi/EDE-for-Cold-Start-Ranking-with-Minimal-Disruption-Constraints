"""Build separate upload documents and figures from verified revision outputs."""
from pathlib import Path
import sys, json, re, copy
import numpy as np
import pandas as pd
sys.path.append(str(Path('revision1/.deps').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path.cwd(); OUT=ROOT/'submitted_peerj_docs'; RESULTS=ROOT/'revision1/results'
OUT.mkdir(parents=True, exist_ok=True)
test=pd.read_csv(RESULTS/'test_summary.csv'); val=pd.read_csv(RESULTS/'validation_summary.csv')
replay=pd.read_csv(RESULTS/'replay/summary.csv'); manifest=json.loads((RESULTS/'replay/manifest.json').read_text())
LAM=json.loads((RESULTS/'protocol.json').read_text())['selected_lambda']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'legend.fontsize':8,'savefig.dpi':300,'figure.dpi':100})
COLORS={'Baseline':'#333333','EDE':'#0072B2','Novelty-only':'#D55E00','Entropy-only':'#009E73','Epsilon':'#CC79A7','Random':'#E69F00'}
MARKERS={'Baseline':'s','EDE':'o','Novelty-only':'^','Entropy-only':'D','Epsilon':'v','Random':'P'}

def select(suite='difficulty',strategy='EDE',penalty=.15,**kwargs):
    frame=test[(test.suite==suite)&(test.strategy==strategy)&np.isclose(test.penalty,penalty)]
    for key,value in kwargs.items(): frame=frame[np.isclose(frame[key],value) if isinstance(value,(int,float)) else frame[key]==value]
    assert len(frame)==1,(suite,strategy,penalty,kwargs,len(frame))
    return frame.iloc[0]

def ms(row,metric,digits=3):
    if metric=='coverage' and digits==3 and row.get('penalty',0)>=.3: digits=4
    if pd.isna(row[metric+'_sd']): return f'{row[metric+"_mean"]:.{digits}f} (n=1; SD NA)'
    return f'{row[metric+"_mean"]:.{digits}f} ± {row[metric+"_sd"]:.{digits}f}'
def ci(row,metric,digits=4): return f'{row[metric]:+.{digits}f} [{row[metric+"_ci_low"]:+.{digits}f}, {row[metric+"_ci_high"]:+.{digits}f}]'
def mean_ci(row,metric,digits=3): return f'{row[metric]:.{digits}f} [{row[metric+"_ci_low"]:.{digits}f}, {row[metric+"_ci_high"]:.{digits}f}]'

def document(title):
    d=Document(); sec=d.sections[0]
    # Remove template title/heading rules inherited from the bundled template.
    for node in list(d.styles.element.iter(qn('w:pBdr'))): node.getparent().remove(node)
    sec.top_margin=sec.bottom_margin=Inches(.8); sec.left_margin=sec.right_margin=Inches(.8)
    sec.page_width=Inches(8.5); sec.page_height=Inches(11)
    normal=d.styles['Normal']; normal.font.name='Times New Roman'; normal.font.size=Pt(11)
    normal.paragraph_format.space_after=Pt(6); normal.paragraph_format.line_spacing=1.12
    for style in ['Title','Heading 1','Heading 2','Heading 3']:
        s=d.styles[style]; s.font.name='Times New Roman'; s.font.color.rgb=RGBColor(0,0,0)
        s.font.size=Pt(16 if style=='Title' else 13 if style=='Heading 1' else 11)
        s.paragraph_format.space_before=Pt(10); s.paragraph_format.space_after=Pt(5)
    d.add_paragraph(title,'Title')
    footer=sec.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    field=OxmlElement('w:fldSimple'); field.set(qn('w:instr'),'PAGE'); footer._p.append(field)
    return d

def p(d,text): return d.add_paragraph(text)
def h(d,text,level=1): return d.add_heading(text,level)
def eq(d,text,number):
    para=d.add_paragraph(); para.alignment=WD_ALIGN_PARAGRAPH.CENTER
    math=OxmlElement('m:oMath'); run=OxmlElement('m:r'); txt=OxmlElement('m:t'); txt.text=text
    run.append(txt); math.append(run); para._p.append(math); para.add_run(f'    ({number})')

def table(d,headers,rows,widths=None):
    tab=d.add_table(rows=1,cols=len(headers)); tab.autofit=False
    if widths is None: widths=[6.9/len(headers)]*len(headers)
    for cell,width in zip(tab.rows[0].cells,widths): cell.width=Inches(width)
    for i,head in enumerate(headers): tab.rows[0].cells[i].text=str(head)
    header_props=tab.rows[0]._tr.get_or_add_trPr(); header_props.append(OxmlElement('w:tblHeader'))
    for row in rows:
        cells=tab.add_row().cells
        for c,value,width in zip(cells,row,widths): c.text=str(value); c.width=Inches(width)
    for ri,row in enumerate(tab.rows):
        trpr=row._tr.get_or_add_trPr(); trpr.append(OxmlElement('w:cantSplit'))
        for cell in row.cells:
            for extra in list(cell.paragraphs)[1:]:
                if not extra.text: extra._p.getparent().remove(extra._p)
            tcpr=cell._tc.get_or_add_tcPr(); borders=OxmlElement('w:tcBorders')
            for edge in ['top','left','bottom','right']:
                el=OxmlElement('w:'+edge); el.set(qn('w:val'),'single'); el.set(qn('w:sz'),'4'); el.set(qn('w:color'),'D9D9D9'); borders.append(el)
            tcpr.append(borders)
            margins=OxmlElement('w:tcMar')
            for edge in ['top','left','bottom','right']:
                el=OxmlElement('w:'+edge); el.set(qn('w:w'),'80'); el.set(qn('w:type'),'dxa'); margins.append(el)
            tcpr.append(margins)
            if ri==0:
                shade=OxmlElement('w:shd'); shade.set(qn('w:fill'),'E5EAF0'); tcpr.append(shade)
            for para in cell.paragraphs:
                para.paragraph_format.space_after=Pt(0); para.paragraph_format.line_spacing=1
                for run in para.runs: run.font.size=Pt(10); run.bold=ri==0
    # Separate adjacent tables: Word otherwise merges them and repeats a wrong header.
    gap=d.add_paragraph(); gap.paragraph_format.space_after=Pt(2); gap.paragraph_format.space_before=Pt(0)
    gap.paragraph_format.line_spacing=Pt(2); gap.add_run().font.size=Pt(1)
    return tab

def save(d,name): d.save(OUT/name)

def figures():
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.2),layout='constrained')
    for st in ['Baseline','EDE','Novelty-only','Entropy-only']:
        frame=test[(test.suite=='difficulty')&(test.strategy==st)].sort_values('penalty')
        for ax,metric in zip(axes,['coverage','ndcg']):
            ax.errorbar(frame.penalty,frame[metric+'_mean'],yerr=frame[metric+'_sd'],color=COLORS[st],marker=MARKERS[st],label=st,capsize=2,linewidth=1.3,markersize=4)
    for ax,label,metric in zip(axes,['(a)','(b)'],['New-document coverage','NDCG@10']):
        ax.set_xlabel('Cold-start penalty p'); ax.set_ylabel(metric); ax.set_title(label,loc='left'); ax.grid(alpha=.15)
    axes[0].legend(); fig.savefig(OUT/'cs-134580-Figure_1.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.2),layout='constrained')
    for ax,pen,label in zip(axes,[.15,.3],['(a)','(b)']):
        f=val[(val.strategy=='EDE')&np.isclose(val.penalty,pen)].sort_values('lam')
        ax.errorbar(f.delta_ndcg,f.delta_coverage,xerr=[f.delta_ndcg-f.delta_ndcg_ci_low,f.delta_ndcg_ci_high-f.delta_ndcg],yerr=[f.delta_coverage-f.delta_coverage_ci_low,f.delta_coverage_ci_high-f.delta_coverage],fmt='o-',color=COLORS['EDE'],capsize=3)
        for j,(_,row) in enumerate(f.iterrows()): ax.annotate(f'λ={row.lam:.1f}',(row.delta_ndcg,row.delta_coverage),xytext=(7,8 if j%2 else -15),textcoords='offset points',fontsize=8)
        ax.axvline(-.01,color='#777777',linestyle='--'); ax.set_xlabel('ΔNDCG@10 versus baseline'); ax.set_ylabel('Δcoverage versus baseline'); ax.set_title(f'{label}  p = {pen:.2f}',loc='left'); ax.grid(alpha=.15)
    fig.savefig(OUT/'cs-134580-Figure_2.png'); plt.close(fig)
    scenarios=['standard','cascade','noisy','strong-bias','heterogeneous','navigational','sparse','delayed','natural']
    fig,axes=plt.subplots(2,1,figsize=(10.4,7),layout='constrained')
    for j,st in enumerate(['Baseline','EDE','Novelty-only','Entropy-only']):
        rows=[select('difficulty' if s=='standard' else 'stress',st,.3,**({} if s=='standard' else {'scenario':s})) for s in scenarios]
        for ax,metric in zip(axes,['coverage','ndcg']):
            ax.errorbar(np.arange(len(scenarios))+(j-1.5)*.12,[r[metric+'_mean'] for r in rows],yerr=[r[metric+'_sd'] for r in rows],fmt=MARKERS[st],color=COLORS[st],label=st,capsize=2,markersize=4)
    for ax,label,metric in zip(axes,['(a)','(b)'],['Coverage','NDCG@10']):
        ax.set_xticks(np.arange(len(scenarios)),scenarios,rotation=20,ha='right'); ax.set_ylabel(metric); ax.set_title(label,loc='left'); ax.grid(axis='y',alpha=.15)
    axes[0].legend(ncol=4); fig.savefig(OUT/'cs-134580-Figure_3.png'); plt.close(fig)
    allrow=replay[replay.criterion=='all'].iloc[0]
    accepted=replay[(replay.criterion!='all')&(replay.accepted==True)]
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.2),layout='constrained')
    axes[0].bar(['Exact list','Same set','Clicks retained'],accepted.acceptance_rate*100,color=['#0072B2','#777777','#999999'])
    for i,rate in enumerate(accepted.acceptance_rate): axes[0].text(i,rate*100+1,f'{rate*100:.2f}%',ha='center',fontsize=9)
    axes[0].set_ylim(0,112); axes[0].set_ylabel('Pages accepted (%)'); axes[0].set_title('(a) Logged-support filters',loc='left')
    metrics=['positions_changed','mean_abs_shift']; x=np.arange(2)
    axes[1].bar(x,[allrow[m] for m in metrics],color=COLORS['EDE'])
    axes[1].errorbar(x,[allrow[m] for m in metrics],yerr=[[allrow[m]-allrow[m+'_ci_low'] for m in metrics],[allrow[m+'_ci_high']-allrow[m] for m in metrics]],fmt='none',color='black',capsize=4)
    axes[1].set_xticks(x,['Positions changed','Absolute rank shift']); axes[1].set_ylabel('Mean per result page'); axes[1].set_title('(b) Displacement diagnostics',loc='left')
    fig.savefig(OUT/'cs-134580-Figure_4.png'); plt.close(fig)

CAPTIONS={
1:f'Figure 1. Corrected simulation discovery and quality across cold-start penalties. (a) Document-level coverage; (b) NDCG@10 over all 2,000 timesteps. Points are means and error bars are sample standard deviations across the same 20 held-out seeds (1000–1019) per strategy and penalty. Each seed has 100 queries, 1,000 documents including 200 new arrivals at timestep 1,000, and one ten-result list per timestep. PBM settings are a=3 and b=0.35. All nonbaseline strategies use λ={LAM:.1f}, η=0.65, m=2, L=50, α=0.5 and τ=50; the baseline has λ=0. Candidate gating and protected prefixes are shared. Coverage means a click on a new document relevant to at least one query; query-matched relevance is reported separately in the supplementary results.',
2:'Figure 2. Validation trade-offs used to choose exploration strength. Panels (a) and (b) show penalties 0.15 and 0.30, respectively. Each point is a paired mean difference from the baseline over ten validation seeds (100–109), with horizontal and vertical 95% Student-t confidence intervals. Labels identify λ in {0.1,0.2,0.3,0.4}; η=0.65, m=2, L=50, α=0.5 and τ=50 are fixed. Each seed uses 2,000 timesteps and the standard PBM simulator. The dashed line marks the −0.01 mean-NDCG selection threshold; it is not a per-query guarantee. These are validation results, not held-out test results or a proven Pareto frontier.',
3:f'Figure 3. Stress tests of (a) coverage and (b) NDCG@10 at penalty 0.30. Points are means and error bars are sample standard deviations over 20 held-out seeds per condition. λ={LAM:.1f}, η=0.65, m=2, L=50, α=0.5 and τ=50 are fixed. Standard, cascade, noisy, stronger position bias, heterogeneous frequency/bias, navigational, sparse-query, delayed-feedback and naturally varying relevance conditions are defined in Section 3.4. Every run has 2,000 timesteps; sparse runs have 1,000 queries, and the other runs have 100. These tests evaluate specified synthetic conditions, not a population of real search intents.',
4:f'Figure 4. Corrected page-level Yandex logged-support diagnostic. The sample contains {manifest["pages"]:,} labeled ten-result pages from the first {manifest["raw_sessions"]:,} complete sessions in train.gz. (a) Exact-list, set-match and click-preservation rates; the latter two are identically 100% because all ten logged candidates are retained, and do not demonstrate cold-start discovery. (b) Mean number of changed positions per page and mean absolute document rank shift, with 95% session-cluster bootstrap confidence intervals (2,000 resamples; seed 20260911); intervals may be narrower than the plotted caps. λ=0.4, η=0.65, m=2, α=0.5 and τ=50; the available candidate count is ten, not 50. Counters use only prior logged events. Reordered click labels are descriptive proxies, not estimates of live engagement.'}


def table_document(title):
    d=document(title)
    sec=d.sections[0]
    sec.top_margin=sec.bottom_margin=sec.left_margin=sec.right_margin=Inches(2.5/2.54)
    d.styles['Title'].font.size=Pt(14)
    d.styles['Normal'].font.size=Pt(10)
    d.styles['Normal'].paragraph_format.line_spacing=1
    return d

def finish_table(d, number, note, size=9):
    global table_notes
    if number==1:
        table_notes=document('Table titles and notes')
    else:
        table_notes.add_page_break()
    p(table_notes,next(para.text for para in d.paragraphs if para.text.strip()))
    p(table_notes,note)
    assert len(d.tables)==1
    t=d.tables[0]
    for row in t.rows:
        for cell in row.cells:
            for margin in cell._tc.iter(qn('w:tcMar')):
                for child in margin:
                    child.set(qn('w:w'),'45' if child.tag in (qn('w:top'),qn('w:bottom')) else '60')
            for para in cell.paragraphs:
                for run in para.runs: run.font.size=Pt(size)
    for para in list(d.paragraphs):
        para._p.getparent().remove(para._p)
    for sec in d.sections:
        for container in (sec.header,sec.footer,sec.first_page_header,sec.first_page_footer,sec.even_page_header,sec.even_page_footer):
            for para in container.paragraphs: para.clear()
    save(d,f'cs-134580-Table{number}.docx')
    if number==6:
        save(table_notes,'cs-134580-Table-titles-and-notes.docx')

def tables():
    common='Test values are mean ± sample SD over 20 held-out seeds (1000–1019). Defaults: λ=0.4, η=0.65, m=2, L=50, α=0.5, τ=50; baseline λ=0. Full per-seed results and paired tests are in the archived CSVs.'
    d=table_document('Table 1 Validation operating points and held-out confirmation')
    rows=[]
    for phase,frame in [('Validation',val),('Held-out',test[test.suite=='difficulty'])]:
        for pen in [.15,.30]:
            for _,r in frame[(frame.strategy=='EDE')&np.isclose(frame.penalty,pen)].sort_values('lam').iterrows():
                rows.append([phase,f'{pen:.2f}',f'{r.lam:.1f}',ci(r,'delta_coverage',3),ci(r,'delta_ndcg',4)])
    table(d,['Sample','Penalty','λ','Δcoverage [95% CI]','ΔNDCG@10 [95% CI]'],rows,[.75,.5,.3,2.4,2.55])
    finish_table(d,1,'Paired differences are EDE minus baseline. Validation uses ten seeds (100–109); held-out confirmation uses 20 disjoint seeds. Student-t 95% CIs are nominal. λ=0.4 maximizes mean validation coverage gain while the lower ΔNDCG bound is at least −0.01 at both penalties. Other defaults are fixed as in Section 3.3. The threshold does not guarantee per-query quality.')

    d=table_document('Table 2 Strategy comparison across difficulty levels')
    groups=[['Baseline','EDE','Novelty-only','Entropy-only'],['Recency','UCB','Thompson','Local-swap'],['Exposure-quota','Random','Epsilon']]
    rows=[]
    for group in groups:
        if rows: rows.append(['Penalty']+group+['']*(4-len(group)))
        for pen in [0.,.10,.15,.20,.25,.30,.40,.60,.80]:
            rows.append([f'{pen:.2f}']+[ms(select(strategy=st,penalty=pen),'coverage') for st in group]+['']*(4-len(group)))
    tab=table(d,['Penalty']+groups[0],rows,[.5,1.5,1.5,1.5,1.5])
    for idx in [10,20]:
        for cell in tab.rows[idx].cells:
            for run in cell.paragraphs[0].runs: run.bold=True
    finish_table(d,2,'Cells show absolute new-document coverage, mean ± SD (20 seeds). The three strategy bands form one table with a shared penalty grid; baseline appears once. All scoring comparators share the prefix and gate. UCB, Thompson, local-swap and quota are defined heuristics, not full published algorithms. Defaults are as in Section 3.3. NDCG, TTF, CTR, unique exposure and all paired tests remain in the archived CSVs.',size=9)

    d=table_document('Table 3 Sensitivity to novelty weight and prefix length')
    rows=[]
    for eta in [.35,.50,.65,.80,1.]:
        for m in [1,2,3]:
            rows.append([f'{eta:.2f}',str(m)]+[ms(select('sensitivity','EDE',pen,eta=eta,m=m),metric,4) for pen in [.15,.30,.40] for metric in ['coverage','ndcg']])
    table(d,['η','m','(a) p=.15\nCoverage','(b) p=.15\nNDCG@10','(c) p=.30\nCoverage','(d) p=.30\nNDCG@10','(e) p=.40\nCoverage','(f) p=.40\nNDCG@10'],rows,[.35,.25]+[.9833]*6)
    finish_table(d,3,'Original Tables 3–6 are consolidated into columns (a)–(d); columns (e)–(f) add p=0.40. Values are mean ± sample SD over 20 held-out seeds. λ=0.4, L=50, α=0.5 and τ=50 are fixed. η and m are varied jointly on the displayed grid; these are sensitivity tests, not further parameter selection.',size=8.5)

    d=table_document('Table 4 Ablations and disruption diagnostics')
    rows=[]
    f=test[(test.suite=='ablation')&np.isclose(test.penalty,.15)].sort_values(['strategy','alpha','m'])
    for _,r in f.iterrows():
        name=r.strategy
        if name=='EDE': name='Unsmoothed α=0' if r.alpha==0 else 'No prefix m=0' if r.m==0 else 'Full EDE'
        r30=select('ablation',r.strategy,.30,alpha=r.alpha,m=r.m)
        rows.append([name,ms(r,'coverage',4),ms(r,'ndcg',4),ms(r30,'coverage',4),ms(r30,'ndcg',4)])
    split=len(rows)+1
    rows.append(['Full-EDE diagnostic','p=0.15','','p=0.30',''])
    metrics=[('rbo','RBO'),('kendall','Kendall distance'),('loss_fraction','Impression loss fraction'),('query_loss_fraction','Query loss fraction'),('worst_query_delta','Worst query ΔNDCG'),('new_discounted_exposure_share','New exposure share'),('promoted_relevance','Promoted relevance'),('promoted_relevant_fraction','Promoted relevant fraction'),('query_relevant_coverage','Query-matched coverage')]
    for metric,label in metrics: rows.append([label,ms(select(penalty=.15),metric),'',ms(select(penalty=.30),metric),''])
    tab=table(d,['Variant / metric','p=.15\nCoverage','p=.15\nNDCG@10','p=.30\nCoverage','p=.30\nNDCG@10'],rows,[1.7,1.2,1.2,1.2,1.2])
    for row in tab.rows[split:]:
        row.cells[1].merge(row.cells[2]); row.cells[3].merge(row.cells[4])
    for cell in tab.rows[split].cells:
        for run in cell.paragraphs[0].runs: run.bold=True
    finish_table(d,4,'Values are mean ± sample SD (20 seeds). Defaults are as in Section 3.3 unless ablated. No decay holds N=1 and also removes its complementary refinement weight. RBO uses persistence 0.9; Kendall uses shared items. Loss fractions exceed 0.01 NDCG loss; worst query is the minimum query-mean delta. Promoted relevance is continuous; promoted relevant fraction is binary and both condition on promotions. New exposure share is discounted item-age exposure, not provider fairness. Additional metrics and p=0.40 diagnostics remain in the archived CSVs.',size=9)

    d=table_document('Table 5 Stress conditions with held-out uncertainty')
    rows=[]
    for scenario in ['cascade','noisy','strong-bias','heterogeneous','navigational','sparse','delayed','natural']:
        for metric in ['coverage','ndcg']:
            rows.append([scenario,'Coverage' if metric=='coverage' else 'NDCG@10']+[ms(select('stress',st,.30,scenario=scenario),metric,4) for st in ['Baseline','EDE','Novelty-only','Entropy-only']])
    table(d,['Condition','Metric','Baseline','EDE','Novelty-only','Entropy-only'],rows,[1.05,.85,1.15,1.15,1.15,1.15])
    finish_table(d,5,'Penalty p=0.30. '+common+' Conditions are defined in Section 3.4. Coverage denominators can differ between conditions. TTF, CTR, unique exposure and additional diagnostics remain in the archived CSVs.',size=9)

    d=table_document('Table 6 Corrected Yandex logged-support diagnostics')
    cols=[('all',None),('exact_match',True),('exact_match',False)]
    selected=[]
    for criterion,accepted in cols:
        f=replay[replay.criterion==criterion]
        if accepted is not None: f=f[f.accepted==accepted]
        selected.append(f.iloc[0])
    rows=[['Pages (count)']+[f'{int(r.pages):,}' for r in selected],['Share of all pages (%)']+[f'{r.pages/manifest["pages"]*100:.3f}' for r in selected]]
    for metric,label,digits in [('positions_changed','Changed positions (count)',3),('mean_abs_shift','Absolute document shift (ranks)',6),('delta_ctr','Observed-label ΔCTR',6),('delta_click_ndcg','Click-NDCG proxy difference',6)]:
        rows.append([label]+[mean_ci(r,metric,digits) for r in selected])
    table(d,['Metric','All pages','Exact-match accepted','Exact-match rejected'],rows,[1.55,1.65,1.65,1.65])
    finish_table(d,6,'Means [95% CI] use 2,000 whole-session bootstrap resamples, seed 20260911. The sample is 372,282 pages from the first 200,000 complete Yandex training sessions. EDE uses λ=0.4, η=0.65, m=2, α=0.5 and τ=50 on ten logged candidates. Set-match and click preservation each accept all 372,282 pages (100%); each has zero rejected pages and undefined rejected means. Click preservation permits changed clicked positions. These are descriptive logged-support diagnostics, not causal engagement or real cold-start discovery estimates. Full clicked-position and accepted/rejected metrics are in the archived replay CSV.',size=9)



if __name__ == "__main__":
    figures()
    tables()
