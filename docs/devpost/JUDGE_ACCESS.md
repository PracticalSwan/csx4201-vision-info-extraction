# Private repository judge access

The official instructions require a private repository to be shared with both:

- `testing@devpost.com`
- `build-week-event@openai.com`

Repository:

<https://github.com/PracticalSwan/csx4201-vision-info-extraction>

In GitHub:

1. Open **Settings → Collaborators and teams**.
2. Choose **Add people**.
3. Enter the first judge email and grant **Read** access.
4. Repeat for the second judge email.
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
