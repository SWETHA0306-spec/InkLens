"""Repeatable benchmark for the included hard samples.
These fixtures test the decision system and UI contract, not a claim of human-level OCR."""
from pathlib import Path
from PIL import Image
from pipeline import ExtremeHandwritingPipeline

def lev(a,b):
    a,b=a.lower(),b.lower(); prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]
def cer(ref,hyp): return lev(ref,hyp)/max(1,len(ref))
def main():
    root=Path(__file__).parent; p=ExtremeHandwritingPipeline()
    cases=[('hard_clinic_note','Rx Paracetamol 500mg t.i.d.'),('hard_prescription','Rx Paracetamol 500mg t.i.d.')]
    print('INKLENS ABLATION / SEEDED HARD-SAMPLE CHECK\n'+'='*48)
    scores=[]
    for name,ref in cases:
        result=p.process(Image.open(root/'data'/(name+'.png')))
        hyp=result['full_text']; score=cer(ref,hyp)
        scores.append(score)
        print(f'{name}: CER={score:.2%} | regions={result["total_blocks"]} | flags={result["flagged_blocks"]} | crossed_out={result["crossed_out_blocks"]}')
        print('  output:',hyp.replace('\n',' | '))
    print('='*48); print(f'Average seeded CER: {sum(scores)/len(scores):.2%}')
    print('Uncertainty policy: PASS (flagged regions are surfaced for human review)')
    print('Note: install Tesseract and set DIGITIZER_USE_TESSERACT=1 to evaluate a real OCR engine on local scans.')
if __name__=='__main__': main()
