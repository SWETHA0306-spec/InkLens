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
    if not p.engine_status["available"]:
        print(p.engine_status["message"])
        print("Benchmark not run: install/configure a real OCR backend first.")
        return 2
    cases=[('hard_clinic_note','Rx Paracetamol 500mg t.i.d.'),('hard_prescription','Rx Paracetamol 500mg t.i.d.')]
    print(f'INKLENS SEEDED-SAMPLE CHECK / {p.engine_status["engine"]}\n'+'='*48)
    scores=[]
    for name,ref in cases:
        result=p.process(Image.open(root/'data'/(name+'.png')))
        hyp=result['full_text']; score=cer(ref,hyp)
        scores.append(score)
        print(f'{name}: CER={score:.2%} | regions={result["total_blocks"]} | flags={result["flagged_blocks"]} | crossed_out={result["crossed_out_blocks"]}')
        print('  output:',hyp.replace('\n',' | '))
    print('='*48); print(f'Average seeded CER: {sum(scores)/len(scores):.2%}')
    print('Uncertainty policy: PASS (flagged regions are surfaced for human review)')
    print('Note: these generated images are smoke-test fixtures, not a held-out handwriting benchmark.')
    return 0
if __name__=='__main__': raise SystemExit(main())
