# API Examples

## Health

```powershell
curl http://127.0.0.1:8001/health
```

## Analyze Text

```powershell
curl -X POST http://127.0.0.1:8001/analyze-text -H "Content-Type: application/json" -d "{\"expected_text\":\"cat\",\"actual_text\":\"cap\",\"accepted_answers\":[\"cat\"],\"debug\":true}"
```

## Analyze Audio

```powershell
curl -X POST http://127.0.0.1:8001/analyze-audio -H "Content-Type: application/json" -d "{\"audio_path\":\"data/samples/sample.wav\",\"expected_text\":\"cat\",\"accepted_answers\":[\"cat\"],\"prompt_id\":\"M2-001\",\"debug\":true}"
```

## Recommend Next

```powershell
curl -X POST http://127.0.0.1:8001/recommend-next -H "Content-Type: application/json" -d "{\"learner_history\":[{\"prompt_id\":\"M2-001\",\"expected_text\":\"cat\",\"actual_text\":\"cap\",\"is_correct\":false,\"error_type\":\"final_sound_error\",\"skill_signal\":\"final_consonant\",\"target_phoneme\":\"T\",\"difficulty_level\":\"easy\"}],\"candidate_items\":[{\"prompt_id\":\"M2-014\",\"expected_text\":\"hat\",\"error_focus\":\"final_consonant\",\"target_phoneme\":\"T\",\"difficulty_level\":\"easy\",\"is_active\":true}],\"top_k\":5}"
```
