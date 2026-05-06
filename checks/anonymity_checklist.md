# Anonymous artifact checklist

Before updating the anonymous repository, scan the repository for identity leaks. Search for your real author names, emails, institutions, lab names, personal usernames, project pages, absolute local paths, and non-anonymous hosting URLs.

A practical command is:

```bash
rg "<replace_with_real_name_or_email_or_institution_or_username>"
```

Repeat the command for each identifying string that could appear in the repository.

## Must remove or anonymize

- Author names, emails, institutions, lab names, and personal usernames.
- GitHub, Hugging Face, website, or project URLs that reveal identity.
- Absolute local paths from a personal machine.
- `CITATION.cff` author fields during double-blind review.
- License copyright holder names if they reveal identity during double-blind review.
- README badges or links to non-anonymous projects.
- Comments or logs revealing identities.

## Must update

- Confirm source dataset license metadata against the original source pages.
- Reconcile the final rewrite model and LLM settings across paper, supplement, code, and metadata where those details are reported.

## Optional cleanup

If older metadata files with rough names are present, remove them before submission to avoid confusing reviewers.
