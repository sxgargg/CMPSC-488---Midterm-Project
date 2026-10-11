#!/usr/bin/env python3
"""Build the report from saved, verified experiment outputs (no invented results)."""
import json,statistics
from pathlib import Path
from xml.sax.saxutils import escape
from datetime import datetime
from zoneinfo import ZoneInfo
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from train_authorship_attribution import ROOT,FAMILIES,load_data

NAVY=colors.HexColor('#163044');TEAL=colors.HexColor('#167D9A');LIGHT=colors.HexColor('#EDF4F7')
PCT=lambda x:f'{100*x:.1f}%'

def main():
    team=json.loads((ROOT/'TEAM.json').read_text());analysis=json.loads((ROOT/'results/analysis.json').read_text())
    runs={p.parent.name:json.loads(p.read_text()) for p in (ROOT/'results').glob('*/metrics.json')}
    assert len(runs)==14,'Run the complete experiment grid before building the report.'
    cnn=runs['cnn_output_s42'];lstm=runs['lstm_output_s42']
    out=ROOT/'output/pdf';out.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=27,leading=31,textColor=NAVY,spaceAfter=18))
    styles.add(ParagraphStyle(name='SectionCustom',fontName='Helvetica-Bold',fontSize=19,leading=24,textColor=NAVY,spaceAfter=13))
    styles.add(ParagraphStyle(name='SubCustom',fontName='Helvetica-Bold',fontSize=11,leading=15,textColor=TEAL,spaceBefore=10,spaceAfter=5))
    styles.add(ParagraphStyle(name='TextCustom',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9,textColor=NAVY))
    styles.add(ParagraphStyle(name='SmallCustom',fontName='Helvetica',fontSize=8.5,leading=11,spaceAfter=6,textColor=NAVY))
    styles.add(ParagraphStyle(name='TableCustom',fontName='Helvetica',fontSize=9,leading=12,textColor=NAVY))
    story=[];md=[]
    def para(text,style='TextCustom'):
        story.append(Paragraph(escape(text),styles[style]));md.append(text+'\n')
    def section(title):
        if story:story.append(PageBreak())
        para(title,'SectionCustom');md.append('')
    def sub(title):para(title,'SubCustom')
    def table(headers,rows,widths=None):
        content=[[Paragraph(escape(str(v)),styles['TableCustom']) for v in row] for row in [headers]+rows]
        t=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),1,TEAL),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#D5E0E5')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        story.extend([t,Spacer(1,10)]);md.extend(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows]+[''])
    def image(name,width=480):
        path=ROOT/'figures'/name;w,h=ImageReader(str(path)).getSize();story.append(Image(str(path),width=width,height=width*h/w));story.append(Spacer(1,8));md.append(f'![{name}](../../figures/{name})\n')
    def ci(r,part='test'):
        a,b=r[part]['accuracy_prompt_bootstrap_95ci'];return f'{PCT(a)} - {PCT(b)}'
    def delta_text(model,tag):
        d=analysis['paired_comparisons'][f'{model}_{tag}'];a,b=d['prompt_bootstrap_95ci'];a=0. if abs(a)<1e-12 else a;b=0. if abs(b)<1e-12 else b;return f'{100*d["difference_b_minus_a"]:+.1f} percentage points (paired 95% interval {100*a:+.1f} to {100*b:+.1f})'

    para('448 PROJECT 1  /  EXPERIMENT REPORT','SubCustom')
    para('Identifying LLM authors\nfrom generated text','TitleCustom')
    para('CNN and LSTM attribution across Llama, Gemini, and Qwen','SubCustom')
    para('Team: '+', '.join(team['members']))
    para('Team leader: '+(team['leader'] or 'Not yet designated (at the team\'s request).'))
    para('Report generated '+datetime.now(ZoneInfo('America/New_York')).strftime('%B %d, %Y')+'.')
    sub('Abstract')
    para(f'We investigate whether small neural classifiers can attribute a response to one of three LLM families using a balanced corpus of 1,500 outputs elicited by 500 shared prompts. A prompt-grouped, category-balanced split prevents the same prompt from appearing in training and evaluation. On the 150-example test set, the output-only CNN achieved {PCT(cnn["test"]["accuracy"])} accuracy and the LSTM achieved {PCT(lstm["test"]["accuracy"])}. Both input-only models scored exactly 33.3%, as expected for identical prompts with balanced labels. We also evaluated input-plus-output models, initialization robustness, held-out writing tasks, and structural-feature ablations. These measurements establish separability within this dataset, not universal or reliable forensic identification of arbitrary text.')
    table(['Research question','Evidence included'],[['RQ1: Which model wrote it?','CNN/LSTM accuracy, macro-F1, per-family recall, confusion matrices, three output-only seeds.'],['RQ2: Does the prompt help?','Input-only, output-only, and input + output under one shared split.'],['RQ3: Unseen tasks?','Four held-out writing categories; source-only checkpoint selection.'],['RQ4: Which signals matter?','Length/formatting summaries, diagnostic baselines, retrained formatting ablations.']],[175,325])
    para('Repository: https://github.com/adithyavemulapalli/448-project-1','SmallCustom')

    section('1  Dataset and provenance')
    table(['Family label','Recorded model','Responses / source'],[['llama','llama3.1:8b','500 / Ollama'],['gemini','gemini-3.8-flash','500 / Gemini API collector'],['qwen','qwen2.5:7b (Q4_K_M)','500 / Ollama on Mac and RTX 4070']],[90,175,235])
    para('The repository master file contains 500 unique prompts: 25 in each of 20 categories. The same prompt text is paired with each family. Every record supplies model_family, model_name, prompt_id, prompt_category, prompt, output, requested temperature/max_tokens, and timestamp. Ollama rows additionally record the endpoint. The repository contains collected generations, not scraped third-party conversations. The original prompt-authoring procedure was not logged; no claim of independently verified prompt originality is made.')
    sub('Coverage')
    para('Categories: explanations; creative fiction; poetry; humor; how-to; recipes; code generation; code debugging; math; riddles; summarization; translation/language; opinions/debate; marketing copy; business writing; history; science; philosophy/ethics; advice; travel.')
    sub('Collection controls and limits')
    para('All records request temperature 0.7 and a 512-token generation cap. This cap does not equalize length across tokenizers or prevent truncation. The Gemini collector requests thinking_budget=0, but thought-token usage was not saved, so actual enforcement cannot be verified. Model names are recorded collection labels; the saved text alone is not independent proof of backend identity.')
    para('Qwen used one verified model digest across the M2 Pro Mac and desktop RTX 4070. Execution moved from local sequential to two local workers, then to remote sequential and finally one worker on each machine. Production had no fixed seed; original saved responses were preserved. Hardware/runtime stages correlate with prompt order, limiting attribution of domain effects. Full provenance is in DATASET_CAVEATS.md and the collection event/metrics sidecars.')
    sub('Audit findings and handling')
    para('All 1,500 records passed unique-ID, prompt/category alignment, nonempty-output, and requested-setting checks. No exact duplicate prompts or outputs were found after whitespace/case normalization; semantic near-duplicates were not exhaustively checked. Qwen response 84 mentions its own name: model names are masked consistently in neural preprocessing. Gemini response 51 contains only two haiku lines and is retained as observed. Of 153 Qwen records with saved finish metadata, 84 reached the cap; this is not a corpus-wide rate. The rejected Pollinations dataset is excluded. Raw data remains unchanged.')

    section('2  Experimental design and implementation')
    para('Main split: 400/50/50 prompt IDs for train/validation/test (1,200/150/150 examples). Each category contributes 20 training prompts; its other five are allocated 2/3 or 3/2 to validation/test, giving 50 each overall. All three responses to a prompt stay together. Splits use seed 42 and are persisted. Vocabulary is built only from the training partition, with minimum token frequency 2.')
    para('Tokenization preserves Unicode words, punctuation, and newlines, then lowercases text and masks model names. Output-only uses the first 256 tokens; input-only uses the first 64 prompt tokens. Input + output concatenates up to 64 prompt tokens, a separator, and the same 256 response tokens. Thus prompts do not displace response text. Unknown words map to a dedicated token. Sequence lengths and padding masks are tracked explicitly.')
    table(['Component','CNN','LSTM'],[['Embedding','Random, 128 dimensions','Random, 128 dimensions'],['Encoder','64 filters per width 3/4/5; ReLU; masked max pooling','One unidirectional layer, 128 hidden units'],['Readout','Concatenated 192-dimensional pooled vector','Hidden state at last real token'],['Head','Dropout 0.4 + three logits','Dropout 0.4 + three logits'],['Output-only parameters',f'{cnn["parameters"]:,}',f'{lstm["parameters"]:,}']],[112,194,194])
    para('Both models use AdamW (learning rate 0.001, weight decay 0.01), cross-entropy loss, batches of 32, gradient norm clipping at 1.0, and up to 15 epochs. Early stopping uses validation loss with patience 4 and minimum improvement 0.0001. A cloned best checkpoint is restored before test evaluation. There was no hyperparameter search or selection on test accuracy. A unidirectional LSTM was chosen so gathering the last real state excludes right padding and runs efficiently on MPS. CNN trailing padded windows cannot win max pooling.')
    para('All 14 experiments ran on the M2 Pro GPU using PyTorch '+cnn['environment']['torch']+'. Output-only seeds 7 and 123 supplement the primary seed 42 with an unchanged split. Test accuracy intervals resample complete prompt triplets (2,000 draws); paired ablation differences use 5,000 draws. These intervals do not capture uncertainty over all model versions or dataset collection procedures. Seed repetitions reuse the same test set.')
    para('Implementation checks cover group separation, category/family balance, Unicode retention, identity masking, matched response budgets, padding invariance, finite gradients, and immutable saved checkpoints. A separate verifier recomputes metrics from predictions, reconstructs vocabularies from training rows, and checks hashes and checkpoint selection.')

    section('3  RQ1: Can responses identify their authors?')
    table(['Model','Accuracy [95% interval]','Macro-F1','Best epoch'],[['CNN',f'{PCT(cnn["test"]["accuracy"])} [{ci(cnn)}]',f'{cnn["test"]["macro_f1"]:.3f}',cnn['best_epoch']],['LSTM',f'{PCT(lstm["test"]["accuracy"])} [{ci(lstm)}]',f'{lstm["test"]["macro_f1"]:.3f}',lstm['best_epoch']]],[65,260,85,90])
    image('confusion_matrices.png',470)
    para('Confusion matrices use true families as rows and predictions as columns; each row contains 50 examples. Per-family accuracy is recall within that family, not precision. All primary models use masked output text.','SmallCustom')
    table(['Model','Gemini recall','Llama recall','Qwen recall'],[[model.upper()]+[PCT(runs[f'{model}_output_s42']['test']['per_family_accuracy'][f]) for f in FAMILIES] for model in ['cnn','lstm']],[100,133,133,134])
    seed_rows=[]
    for model in ['cnn','lstm']:
        values=[runs[f'{model}_output_s{s}']['test']['accuracy'] for s in [42,7,123]]
        seed_rows.append([model.upper()]+[PCT(x) for x in values]+[f'{100*statistics.mean(values):.1f} +/- {100*statistics.stdev(values):.1f}'])
    table(['Model','Seed 42','Seed 7','Seed 123','Mean +/- SD (pp)'],seed_rows,[65,85,85,85,180])
    para('Performance above the 33.3% balanced baseline supports attribution within this collection. Differences between architectures may reflect local formatting cues, how information is pooled, and optimization, not an intrinsic advantage on every authorship problem. The repeated seeds measure initialization/training variability only. No seed was selected retrospectively as the headline result.')

    section('4  RQ2: Does the user prompt help?')
    table(['Model','Text source','Accuracy','Macro-F1'],[[m.upper(),s,PCT(runs[f'{m}_{s}_s42']['test']['accuracy']),f'{runs[f"{m}_{s}_s42"]["test"]["macro_f1"]:.3f}'] for m in ['cnn','lstm'] for s in ['input','output','both']],[75,180,120,125])
    image('rq2_accuracy.png',465)
    para('Input-only accuracy is exactly one third for both architectures. Each unseen prompt occurs three times with identical text and three different balanced labels; a deterministic model makes one prediction for all three, so exactly one is correct. This is stronger than a statistical expectation of chance and acts as a pipeline check. Input-only macro-F1 can differ from one third because the model can favor some labels.')
    para('Adding prompts changes CNN accuracy by '+delta_text('cnn','adding_prompt')+' and LSTM accuracy by '+delta_text('lstm','adding_prompt')+'. The comparison uses paired prompt groups. A range crossing zero does not support a clear improvement from this test sample.')
    para('Identical prompts cannot reveal the family by themselves, but may help interpret response style conditional on the task. Including them also lengthens the sequence and changes vocabulary/optimization. Consequently, either improvement or degradation is possible; output-only performance is not a theoretical upper bound.')

    section('5  RQ3: Do fingerprints transfer across tasks?')
    para('This extra-credit experiment excludes creative fiction, poetry, marketing copy, and business writing from training and validation. The remaining 16 categories provide 320 training, 40 validation, and 40 source-test prompts (960/120/120 examples). The held-out writing set contains 100 prompts and 300 responses. Each model is trained from scratch on source output text only, seed 42. Vocabulary and checkpoint selection use source data exclusively.')
    domain_rows=[]
    for m in ['cnn','lstm']:
        r=runs[f'{m}_domain_s42'];domain_rows.append([m.upper(),PCT(r['test']['accuracy']),PCT(r['target']['accuracy']),f'{100*(r["target"]["accuracy"]-r["test"]["accuracy"]):+.1f}',f'{r["target"]["macro_f1"]:.3f}'])
    table(['Model','Source test','Unseen writing','Change (pp)','Target macro-F1'],domain_rows,[65,100,115,100,120])
    table(['Model','Target Gemini','Target Llama','Target Qwen','Target accuracy 95% CI'],[[m.upper()]+[PCT(runs[f'{m}_domain_s42']['target']['per_family_accuracy'][f]) for f in FAMILIES]+[ci(runs[f'{m}_domain_s42'],'target')] for m in ['cnn','lstm']],[65,95,95,95,150])
    para('The target test set is wholly unseen by task category, and therefore tests transfer beyond new prompts in familiar categories. Source and target samples differ in size and difficulty. Their accuracy difference is descriptive: it does not isolate domain as a causal variable. The source training set is also smaller than in RQ1, so domain results should not be compared directly with the primary test as if training conditions were identical.')
    para('Both architectures lose accuracy on unseen writing in these runs. Their above-chance target performance is accompanied by substantial losses, limiting claims of task-independent fingerprints.')
    para('Above-chance target performance is consistent with some transferable signals; a decline indicates that part of the learned evidence depends on task mix. It does not prove that any signal is universal to an LLM family. Collection stages, response lengths, and formatting may all contribute. Only four writing categories and one domain-training seed per architecture were tested.')
    sub('Per-family interpretation')
    for m in ['cnn','lstm']:
        r=runs[f'{m}_domain_s42'];v=r['target']['per_family_accuracy'];best=max(v,key=v.get);worst=min(v,key=v.get)
        para(f'{m.upper()}: target recall is highest for {best} ({PCT(v[best])}) and lowest for {worst} ({PCT(v[worst])}). This difference reflects the particular held-out writing responses; it is not an intrinsic identifiability ranking for all future models.')

    section('6  RQ4: What signals distinguish the outputs?')
    image('style_features.png',485)
    para('Distributions describe all 500 responses per family; boxplots hide outliers for readability. Words are Unicode letter/number spans. Lexical diversity is unique-word count divided by word count, so it is length-dependent. These summaries are descriptive and not causal feature attributions.','SmallCustom')
    base=analysis['diagnostic_baselines']
    table(['Diagnostic model','Test accuracy','Macro-F1'],[['Word count only',PCT(base['length_only']['accuracy']),f'{base["length_only"]["macro_f1"]:.3f}'],['11 structural/lexical features',PCT(base['structure_all']['accuracy']),f'{base["structure_all"]["macro_f1"]:.3f}']],[270,115,115])
    para('Diagnostic models use standardized logistic regression fitted only on training data. Features are words, characters, lines, blank lines, bullet lines, heading lines, backticks, bold markers, type-token ratio, average word length, and punctuation fraction. They access complete responses, while neural models use capped sequences; they are not exactly input-matched competitors.')
    table(['Model','Masked output','Plain words','Difference (pp)'],[[m.upper(),PCT(runs[f'{m}_output_s42']['test']['accuracy']),PCT(runs[f'{m}_plain_s42']['test']['accuracy']),f'{100*(runs[f"{m}_plain_s42"]["test"]["accuracy"]-runs[f"{m}_output_s42"]["test"]["accuracy"]):+.1f}'] for m in ['cnn','lstm']],[80,145,145,130])
    para('Retraining with punctuation/newlines removed changes CNN accuracy by '+delta_text('cnn','removing_formatting')+' and LSTM accuracy by '+delta_text('lstm','removing_formatting')+'. Both variants already lowercase and mask model names. Removing formatting also allows more words inside the fixed token budget, so the comparison cannot attribute the entire difference solely to punctuation.')
    para(f'Word count alone scored {PCT(base["length_only"]["accuracy"])}, whereas the combined structural/lexical features scored {PCT(base["structure_all"]["accuracy"])}. Both neural models lost accuracy after formatting removal. This pattern is consistent with substantial structural information beyond length alone, with the token-budget confound noted above.')

    section('7  Limitations, lessons, and experience')
    sub('What this study does and does not establish')
    para('This is a closed-set classification study of three recorded model configurations and one generation setup per family. It does not detect human authorship, handle unknown families, or establish reliability on paraphrased, edited, adversarial, or newly released model outputs. A family label may proxy for the specific model, provider, chat template, formatting defaults, or collection stage. The dataset contains 500 independent prompt groups, not 1,500 independent tasks.')
    para('The 512 generation cap leaves some responses incomplete, and the 256-token neural budget further removes text. Short answers and incomplete generations remain in the raw corpus. Missing Gemini finish/thought metadata prevents resolving whether its unusually short haiku reflects normal generation, truncation, or another cause. Similarities between different prompts were not exhaustively audited. Test intervals and initialization repeats cannot compensate for these provenance limits.')
    sub('Technical lessons from the project workflow')
    para('1. Data integrity precedes model complexity. Explicit file selection, family balance, prompt alignment, and checksums prevented silent omissions such as ignoring a JSONL family. The input-only control provides a concrete check against accidental label or metadata leakage.')
    para('2. Padding and checkpoint details change what is actually evaluated. A unidirectional LSTM must read the last real token rather than padded timesteps; CNN pooling must exclude trailing padding. Cloning the saved best state prevents later optimization from mutating the intended checkpoint.')
    para('3. Equal sampling parameters are useful controls, but do not equalize distributions. Different tokenizers, response lengths, formatting, runtime versions, and execution stages remain possible shortcuts. The length/structure baselines and formatting ablations make some of these visible.')
    para('4. Faster hardware improves iteration, but integrity matters during migration. Checkpointing and one writer preserved all saved responses while Qwen collection moved between Mac and PC. The training suite keeps splits and outputs separate for each run, making comparisons reviewable.')
    para('5. A successful classifier is not automatically an explanation. Domain transfer and ablations narrow interpretations, but neither proves a universal fingerprint. Reporting negative controls, confidence intervals, and weaker configurations is more informative than reporting only the best seed.')
    sub('Assistance and responsibility')
    para('Codex assisted with collection tooling, auditing, implementation, experiment execution, analysis, and report preparation. These lessons describe the documented workflow rather than invented personal reflections. The named team should review the code, results, report, and course disclosure requirements and be able to explain the submission. The team leader is intentionally not designated yet.')

    section('8  Reproducibility and references')
    para('Repository: https://github.com/adithyavemulapalli/448-project-1. Raw dataset checksums and collection history are versioned. Each experiment stores configuration, environment, model parameter count, training history, best epoch, data hashes, vocabulary, prompt split, and validation/test predictions. The two primary output-only checkpoints, including weights and vocabulary, are versioned in checkpoints/. Other run checkpoints are saved locally in models/ and can be reproduced by running the suite.')
    sub('Execution sequence')
    for command in ['python3 -m venv .venv', 'source .venv/bin/activate', 'pip install -r requirements.txt', 'python -m unittest discover -s tests -v', 'python run_experiments.py --device auto', 'python analyze_results.py', 'python verify_results.py', 'python build_report.py']:
        para(command,'SmallCustom')
    para('Use --device cuda on a CUDA-enabled PyTorch installation or --device mps on Apple Silicon. CPU is supported. requirements-lock.txt captures the actual Mac environment; GPU backend differences may prevent bit-for-bit reproduction. Individual runs accept --model cnn|lstm and --text-source input|output|both. Existing completed runs are retained rather than silently overwritten.')
    seconds=sum(r['training_seconds'] for r in runs.values())
    para(f'The recorded training and per-epoch validation loops across 14 runs total {seconds:.1f} seconds ({seconds/60:.1f} minutes). This excludes process startup, preprocessing, final prediction export, plotting, and report generation; it is not end-to-end elapsed time.')
    sub('References')
    refs=[
    '[1] Y. Kim. Convolutional Neural Networks for Sentence Classification. EMNLP, 2014, pp. 1746-1751. https://aclanthology.org/D14-1181/ (CNN architecture background; our embeddings are trained from scratch.)',
    '[2] S. Hochreiter and J. Schmidhuber. Long Short-Term Memory. Neural Computation 9(8), 1997, pp. 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735',
    '[3] PyTorch documentation. torch.nn.LSTM. https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html',
    '[4] Scikit-learn documentation. LogisticRegression and StandardScaler. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html',
    '[5] Project artifacts: DATASET_CAVEATS.md, EXPERIMENTS.md, DATASET_STATUS.json, and results/*/metrics.json. Primary evidence for dataset and reported experimental claims.'
    ]
    for ref in refs:para(ref,'SmallCustom')
    para('Submission item still requiring a team decision: designate the team leader. No leader was selected by the assistant.','SmallCustom')
    def page(canvas,doc):
        canvas.saveState();w,h=doc.pagesize;canvas.setStrokeColor(TEAL);canvas.setLineWidth(1);canvas.line(48,h-42,w-48,h-42);canvas.setFillColor(NAVY);canvas.setFont('Helvetica',8);canvas.drawString(48,h-33,'448 PROJECT 1  |  LLM AUTHORSHIP ATTRIBUTION');canvas.drawString(48,28,'Adithya Vemulapali / Shreya Garg / Ani Ganesan');canvas.drawRightString(w-48,28,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(out/'448_project_1_report.pdf'),pagesize=(612,792),leftMargin=56,rightMargin=56,topMargin=62,bottomMargin=48,title='448 Project 1: LLM Authorship Attribution',author=', '.join(team['members']))
    doc.build(story,onFirstPage=page,onLaterPages=page)
    (out/'448_project_1_report.md').write_text('\n'.join(md).rstrip()+'\n')
    print(out/'448_project_1_report.pdf')
if __name__=='__main__':main()
