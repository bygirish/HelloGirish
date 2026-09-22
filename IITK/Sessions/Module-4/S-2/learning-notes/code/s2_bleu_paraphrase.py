import sacrebleu
refs = [["a teacher is speaking to students"]]
for cand in ["agentic AI is being taught", "a teacher is talking to students", "a teacher is speaking to students"]:
    b = sacrebleu.corpus_bleu([cand], refs)
    print(f"BLEU={b.score:6.2f}  precisions={[round(p,1) for p in b.precisions]}  <- {cand!r}")
