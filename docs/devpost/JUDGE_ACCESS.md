# Repository visibility and judge access

Repository:

<https://github.com/PracticalSwan/csx4201-vision-info-extraction>

## Current state

The owner intentionally made the repository public on 2026-07-21. Judges and
other readers can access the source, README, MIT license, contribution policy,
and published Release without a GitHub invitation.

Earlier private-repository judge invitations are no longer required for read
access. Do not describe the repository or Release as private in current
submission materials.

Public visibility does not weaken the data boundary:

- raw/private Gmail documents remain ignored and must never be uploaded;
- private OCR text, filenames, images, and per-document predictions remain
  local;
- `runtime.local.json`, `.runtime`, outputs, caches, and credentials remain
  excluded; and
- the public Release is built from an allowlist and must pass its privacy
  audit before upload.

Before any future visibility, access, or Release change, verify:

```powershell
gh repo view PracticalSwan/csx4201-vision-info-extraction `
  --json visibility,url
git status --short
git ls-files | rg -i "gmail|private|\.env|token|secret|password"
```

Repository access never authorizes uploading private source or derived data.
