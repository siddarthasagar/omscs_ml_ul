# LaTeX Float Placement Guide
> Findings from research + applied fixes on UL_Report_schinne3.tex (April 2026)

---

## The Core Problem

LaTeX's float system places tables and figures at the "best" typographic position,
which often means they drift away from where they are declared in source — sometimes
into the wrong section entirely.

---

## Placement Specifiers

| Specifier | Meaning | Notes |
|-----------|---------|-------|
| `[t]` | Top of column/page | IEEE default; reliable but drifts freely |
| `[b]` | Bottom of column/page | Secondary fallback |
| `[h]` | Here (inline) | Soft preference; often ignored in two-column |
| `[!h]` / `[!ht]` | Force relaxed constraints, try here first | Best general-purpose choice |
| `[H]` | Exact position, non-floating (requires `float` pkg) | Last resort; breaks in IEEE two-column |

**Rule of thumb for IEEE two-column:** Use `[!ht]` on all tables and figures.
Never use `[H]` — it interacts poorly with column-width constraints.

---

## Preventing Drift Across Sections

### Option A — Auto-barrier at every section (recommended)

```latex
\usepackage[section]{placeins}
```

This inserts a `\FloatBarrier` automatically at every `\section` command.
Tables and figures can no longer float past the section they are declared in.

### Option B — Manual barrier at specific points

```latex
\FloatBarrier   % floats above this line must be placed before continuing
\section{Next Section}
```

Use this when you need finer control than section-level (e.g., between subsections).

### Option C — Flush all pending floats (nuclear option)

```latex
\clearpage
```

Forces all pending floats to be placed immediately, then starts a new page.
Avoid this in two-column papers — it wastes space.

---

## Tables That Should Appear Right After a Paragraph

```latex
Some paragraph text here.

\FloatBarrier   % prevent earlier floats from pushing past
\begin{table}[!ht]
  \centering
  \caption{Your caption}
  ...
\end{table}
```

---

## Figures That Must Not Float Before Their Section Heading

Change the figure specifier from `[t]` to `[h!]`:

```latex
\begin{figure}[h!]
  ...
\end{figure}
```

Used for figures like t-SNE that appear at the start of their own section and
must not float ahead of the section heading.

---

## Float Counter Tuning (already set in UL report preamble)

If floats still stack up, tune these parameters:

```latex
\renewcommand{\topfraction}{0.9}       % max fraction of page for top floats
\renewcommand{\bottomfraction}{0.9}    % max fraction of page for bottom floats
\renewcommand{\textfraction}{0.05}     % min fraction of page that must be text
\renewcommand{\floatpagefraction}{0.85}% min float fill on a float-only page
\setcounter{topnumber}{4}              % max floats at top of a page
\setcounter{bottomnumber}{3}
\setcounter{totalnumber}{8}
\setcounter{dbltopnumber}{3}           % max floats at top in two-column mode
```

---

## Two-Column Specific Notes

- `table` (single-column width) — use `[!ht]`
- `table*` (full page width, spans both columns) — use `[!t]`
- Add `\usepackage{dblfloatfix}` if two-column floats appear out of order

---

## Summary: What Was Applied to UL Report

| Problem | Fix Applied |
|---------|-------------|
| Tables drifting into wrong sections | `\usepackage[section]{placeins}` |
| All table specifiers too permissive | Changed `[t]` → `[!ht]` on all 5 tables |
| t-SNE figure appearing before its section heading | Changed `[t]` → `[h!]` |
| Step 2/3/4-5 floats bleeding into next section | Added `\FloatBarrier` before each `\section` |
