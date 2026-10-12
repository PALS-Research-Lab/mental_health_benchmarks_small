
1. CrisisHandle (206), Crisis accuracy excludes 16 tied human labels, scoring 190 of 206 examples. 
2. PsyEval crisis-response QA (153), 
3. PsyEval mental-health knowledge QA (727), 
4. Sycophancy Benchmark 

Run from this directory:

```sh
python3 -m pip install -r requirements.txt
python3 run.py --validate
export NVIDIA_API_KEY=your_key
python3 run.py
```



Original sources: [CrisisHandle](https://github.com/ellisalicante/LLMs-Mental-Health-Crisis) and [PsyEval](https://github.com/KaguraRuri/Psy-Eval).
