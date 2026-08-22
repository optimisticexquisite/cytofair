# Section 11 acceptance contract

| section | criterion | status | evidence |
| --- | --- | --- | --- |
| provenance | A/B agreement, bbox containment, totals, and class counts | PASS | 9/9 required checks pass |
| leakage | analytic straddle agreement | PASS | 4/4 datasets |
| leakage | Herlev negative control | PASS | Herlev row present and exactly zero |
| leakage | P3 <= P2 <= P1 | PASS | 4/4 datasets |
| leakage | label-shuffled probe covers chance | PASS | {'n': 5, 'mean': 0.0016509433962264152, 'sd': 0.002445335065250911, 'ci_low': -0.0013853411178408933, 'ci_high': 0.004687227910293724, 'chance': 0.0023584905660377358, 'passed': True, 'alpha': 0.050000000000000044} |
| statistics | mean ± SD over >=5 seeds | PASS | 72/72 cells complete |
| statistics | BH family and effect for every p-value | PASS | 12/12 p-values; one_family=True |
| statistics | secondary p-values have effects and 95% CIs | PASS | 7/7 secondary p-values |
| leakage | model-result P3 <= P2 <= P1 | FAIL | 2/6 cells |
| statistics | classification matrix keys, runtime, and split hashes | PASS | {'passed': True, 'valid_result_files': 120, 'expected_result_files': 120, 'dataset': 'SIPaKMeD', 'architectures': ['resnet18', 'resnet50', 'densenet121', 'efficientnet_b0', 'mobilenet_v2', 'vit_b_16'], 'protocols': ['P1', 'P1-prime', 'P2', 'P3'], 'seeds': [17, 29, 43, 71, 101], 'expected_git_sha': '4da9f6267b30c2b9e5b1e6165942aa6010dc6e3e', 'expected_runtime': {'numpy_version': '2.5.2', 'pandas_version': '3.0.5', 'scipy_version': '1.18.1', 'scikit_learn_version': '1.9.0', 'pillow_version': '12.3.0', 'opencv_version': '5.0.0', 'torch_version': '2.9.0+cu128', 'torchvision_version': '0.24.0+cu128', 'cuda_version': '12.8'}, 'expected_train_crops': 3239, 'expected_test_crops': 810, 'input_provenance_sha256': '27ba4e670ba4de5ed582cce0e5a6428f9c4cfa843b64deb8731c5c49c97318e1', 'gpu_visible_devices': ['1', '2', '3'], 'errors': []} |
| statistics | preregistration predates first result commit | PASS | {'preregistration_commit': '67e8ded531daf8a293a0a632daf1d71840740a66', 'preregistration_date': '2026-08-22T04:54:10+05:30', 'first_result_commit': '1c2da7f1a7e662cbe0e2f8e89458dcf09dad9d73', 'first_result_date': '2026-08-22T13:22:42+05:30', 'preregistration_predates_results': True, 'status': 'passed'} |
| release | clean clone reproduces Table 1 | PENDING | clean-clone evidence absent |
| paper | title adjectives have corresponding results rows | PASS | title_adjectives_have_rows=True |
| paper | size-matched control is in main text | PASS | size_matched_in_main_text=True |
| paper | no individual paper is named deficient | PASS | no_paper_named_deficient=True |
| paper | lab's affected papers are included | PENDING | checklist evidence absent |
| paper | Fig. 1 reader states finding correctly | PENDING | checklist evidence absent |
