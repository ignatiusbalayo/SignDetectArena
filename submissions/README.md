# Submissions

Participants add exactly one folder here through a Pull Request:

```
submissions/inbox/<team>/submission.enc   # your encrypted predictions
submissions/inbox/<team>/metadata.json    # team / model / type / notes
```

Create both with `python scripts/encrypt_submission.py` (see the main README).
The scoring bot decrypts, scores and updates the leaderboard automatically.
Do not edit anything else in the repository in a submission PR - it will be rejected.
