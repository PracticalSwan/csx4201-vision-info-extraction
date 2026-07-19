# Private repository judge access

The official instructions require a private repository to be shared with both:

- `testing@devpost.com`
- `build-week-event@openai.com`

Repository:

<https://github.com/PracticalSwan/csx4201-vision-info-extraction>

## Current access state

- `testing@devpost.com`: active as the email-resolved GitHub account
  `devposttesting`, with pull-only **Read** access. The original GitHub
  invitation ID was `326151362`.
- `build-week-event@openai.com`: pending email invitation `326199273`, with
  **Read** permission, created on 2026-07-19. GitHub initially created the
  email invitation with `write`; it was immediately reduced to `read` and
  reverified through the authenticated GitHub API.

To verify the current state:

```powershell
gh api repos/PracticalSwan/csx4201-vision-info-extraction/collaborators `
  --jq '.[] | {login,permissions}'
gh api repos/PracticalSwan/csx4201-vision-info-extraction/invitations `
  --jq '.[] | {id,permissions,created_at}'
```

The expected result is `devposttesting` with `pull: true` and `push: false`,
plus pending invitation `326199273` with `permissions: "read"`.

If GitHub does not resolve an email to an account, use the exact alternative
judge-access mechanism stated in the current Devpost FAQ or contact the event
organizers. Do not make the repository public as a workaround: its history and
project documentation assume a private repository, and private dataset guards
must still be audited before any visibility change.

Before changing judge access, verify:

```powershell
git status --short
git ls-files | rg -i "gmail|private|\.env|token|secret|password"
```

Judge access does not authorize uploading raw private data, private filenames,
private outputs, or local environment files.
