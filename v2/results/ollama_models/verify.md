# Ollama song/ 모델 확인

서버 OLLAMA_NOPRUNE:true 확인: True

| 모델 | 상태 | 주 GGUF 블롭 | 크기 | 기대값 출처 | 확인 항목 | 짧은 응답 (모드) |
|---|---|---|---:|---|---|---|
| `song/gemma4-26b-cafe-98e` | registered OK | `061cc70f4814` | 13.67 GB | results/step5/overnight_notes.md | main_blob_expected ✓, main_blob_file_exists ✓, projector_and_draft ✓, blob_content_matches_name ✓, prompt_ok ✓ | 안녕하세요, 만나서 반갑습니다! (18.1 s, igpu) |
| `song/gemma4-26b-cafe-64e` | registered OK | `1fa88d73cd3f` | 9.81 GB | scripts/report_step6.py BLOB | main_blob_expected ✓, main_blob_file_exists ✓, projector_and_draft ✓, blob_content_matches_name ✓, prompt_ok ✓ | 안녕하세요, 만나서 반갑습니다! (10.3 s, igpu) |
| `song/gemma4-26b-cafe-48e` | registered OK | `9a01e444a066` | 8.00 GB | scripts/report_step6.py BLOB | main_blob_expected ✓, main_blob_file_exists ✓, projector_and_draft ✓, blob_content_matches_name ✓, prompt_ok ✓ | 반갑습니다, 만나서 정말 기뻐요! (10.1 s, igpu) |
| `song/gemma4-26b-cafe-64e-text` | registered OK | `1fa88d73cd3f` | 9.81 GB | models/gguf/lean/Modelfile.64e-text FROM | main_blob_expected ✓, main_blob_file_exists ✓, text_only_no_projector ✓, blob_content_matches_name ✓, prompt_ok ✓ | 안녕하세요, 만나서 반갑습니다! (10.2 s, igpu) |
| `song/gemma4-26b-cafe-48e-text` | registered OK | `9a01e444a066` | 8.00 GB | models/gguf/lean/Modelfile.48e-text FROM | main_blob_expected ✓, main_blob_file_exists ✓, text_only_no_projector ✓, blob_content_matches_name ✓, prompt_ok ✓ | 반갑습니다, 만나서 정말 기뻐요! (9.3 s, igpu) |
| `song/gemma4-26b-128e-noimat` | registered OK | `caf0cbfc0553` | 17.07 GB | no hash recorded; logs/overnight_step5.log: registered 18:24:35 'blob sha256 == local Q4', local Q4 (17.1 GB) deleted 18:24:37 | main_blob_expected ✓, main_blob_file_exists ✓, projector_and_draft ✓, blob_content_matches_name ✓, prompt_ok ✓ | 안녕하세요, 만나서 반갑습니다! (18.6 s, cpu) |
| `song/gemma4-26b-cafe-98e-reap` | registered OK | `0da089756251` | 13.67 GB | v2/results/reap_curve/reap98/pipeline_state.json | main_blob_expected ✓, main_blob_file_exists ✓, projector_and_draft ✓, blob_content_matches_name ✓, prompt_ok ✓ | 안녕하세요, 만나서 반갑습니다! (19.1 s, igpu) |
| `song/gemma4-26b-cafe-64e-reap` | registered OK | `b92b9962bec4` | 9.81 GB | v2/results/reap64/pipeline_state.json | main_blob_expected ✓, main_blob_file_exists ✓, projector_and_draft ✓, blob_content_matches_name ✓, prompt_ok ✓ | 안녕하세요! 만나서 정말 반갑습니다.   오늘 하루도  (10.6 s, igpu) |
| `song/gemma4-26b-cafe-48e-reap` | registered OK | `a8832fd38f80` | 8.00 GB | v2/results/reap_curve/reap48/pipeline_state.json | main_blob_expected ✓, main_blob_file_exists ✓, projector_and_draft ✓, blob_content_matches_name ✓, prompt_ok ✓ | 안녕하세요, 만나서 반갑습니다! (9.4 s, igpu) |

- 128e-noimat은 v1처럼 CPU 전용(`num_gpu 0`)으로 확인했다. iGPU로는 Vulkan 버퍼 할당 실패(메모리 부족)로 뜨지 않는다.
