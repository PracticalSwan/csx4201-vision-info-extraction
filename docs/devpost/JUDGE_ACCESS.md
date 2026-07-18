# Private repository judge access

The official instructions require a private repository to be shared with both:

- `testing@devpost.com`
- `build-week-event@openai.com`

Repository:

<https://github.com/PracticalSwan/csx4201-vision-info-extraction>

## Current access state

- `testing@devpost.com`: invitation created on 2026-07-19 for the
  email-resolved GitHub account `devposttesting`, with **Read** permission.
  GitHub invitation ID: `326151362`.
- `build-week-event@openai.com`: pending. GitHub's public user search does not
  expose an account for this email, and the repository settings page requires
  the owner to complete passkey/sudo confirmation before GitHub will resolve
  the address. Do not substitute the similarly named public organization
  `openai-build-week`; it is not verified as the official judge account.

In GitHub:

1. Open **Settings → Collaborators and teams**.
2. Choose **Add people**.
3. Complete GitHub's passkey/sudo confirmation if prompted.
4. Enter `build-week-event@openai.com` and grant **Read** access.
5. Confirm both invitations are visible before final submission.

If GitHub does not resolve an email to an account, use the exact alternative
judge-access mechanism stated in the current Devpost FAQ or contact the event
organizers. Do not make the repository public as a workaround: its history and
project documentation assume a private repository, and private dataset guards
must still be audited before any visibility change.

Before inviting judges, verify:

```powershell
git status --short
git ls-files | rg -i "gmail|private|\.env|token|secret|password"
```

Judge access does not authorize uploading raw private data, private filenames,
private outputs, or local environment files.
