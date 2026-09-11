# Commit hash map

On 2026-09-10 the authorship of the 18 commits after `dd2ab09` was reassigned across the team.
Only author and committer identity changed. Every tree and message is byte-identical, so the code and
results are unchanged, but those commits got new hashes. Results files record the hash of the commit
they were measured on; use this table to find the current commit for a recorded hash.

| # | Recorded (old) hash | Current hash | Subject |
|---|---|---|---|
| 1 | `d39b6fe` | `d39b6fe` | README: what it is, how to run it, and every measured number with its caveats |
| 2 | `fa36a2e` | `db38a74` | Stage 1B: measure the two signals that don't work, and retract a vacuous 0% |
| 3 | `8875049` | `b06c3d5` | PoisonedRAG: scale the replay to n=50/10k, and record that the result is invalid |
| 4 | `8b3ae62` | `4664642` | Cite TrustRAG properly, and the paper that measures its clean-accuracy cost |
| 5 | `650ef8f` | `e2756c0` | Record two claims we cannot source, so neither reaches a slide |
| 6 | `174de52` | `b6f3e8e` | PoisonedRAG headline: ASR 62%->47% at n=100/10k, clean accuracy unchanged |
| 7 | `229b1b0` | `6146dcc` | Drop the perplexity baseline from scope |
| 8 | `2039e3d` | `1f8f8c4` | BIPIA held-out eval: Stage 1A catches 0 of 41,250 attacked documents |
| 9 | `eba3bb4` | `bd222b8` | README: report the BIPIA zero as a pre-registered test that failed |
| 10 | `b115de2` | `3438a50` | README: record the attack class neither Stage 1A nor Stage 3 covers |
| 11 | `23238fe` | `13ebbe0` | Stage 3 measured: the URL checker has no denominator here, and the tool guard has no headroom |
| 12 | `0e88e7d` | `919c897` | README: stage3/ is no longer paused in the layout map |
| 13 | `2dffc36` | `2726654` | Wire the API to the real pipeline, and make --real refuse to lie |
| 14 | `2ff94c3` | `ec0a515` | Fix the collapse wiring, and measure that fixing it changes ASR by nothing |
| 15 | `e365628` | `cc4c585` | README: report what collapse actually buys, and point the UI at the corrected run |
| 16 | `9339419` | `7a44e81` | Make the live probe run the experiment it claims, and stop the trace contradicting itself |
| 17 | `83568bb` | `c8a8045` | Retune ingestion, add batch_cluster, hygiene and a context firewall: ASR 62% -> 12% |
| 18 | `14504cf` | `d890393` | README: make the held-out 50 the ASR headline and mark what is stale |

## Results files by recorded hash

| Recorded hash | Current hash | Results files |
|---|---|---|
| `229b1b0` | `6146dcc` | bipia_20260910T164029Z.json |
| `2dffc36` | `2726654` | poisonedrag_n100_20260910T195905Z.json |
| `5ae7fb7` | `5ae7fb7` (pushed commit, not rewritten) | tenant_leak_20260910T112318Z.json |
| `650ef8f` | `e2756c0` | poisonedrag_n100_20260910T153530Z.json |
| `9339419` | `7a44e81` | poisonedrag_n100_20260910T213715Z.json, poisonedrag_n100_20260910T220208Z.json, poisonedrag_n100_20260910T220602Z.json, poisonedrag_n100_20260910T224217Z.json |
| `b115de2` | `3438a50` | egress_20260910T190656Z.json |
| `d39b6fe` | `d39b6fe` | geometry_20260910T142941Z.json, poisonedrag_n10_20260910T142421Z.json, poisonedrag_n10_20260910T142424Z.json |
| `f566c97` | `f566c97` (pushed commit, not rewritten) | injection_20260910T120229Z.json, injection_20260910T121109Z.json, poisonedrag_n2_20260910T113806Z.json, tenant_leak_20260910T120126Z.json |
| `fa36a2e` | `db38a74` | poisonedrag_n50_20260910T143808Z.json |
