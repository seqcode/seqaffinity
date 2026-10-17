# Generate cross-check numbers from published R implementations.
#
# NOT RUN AS PART OF THE TEST SUITE. Needs Bioconductor (MatrixRider, TFBSTools,
# Biostrings) and tRap (github.com/matthiasheinig/tRap). Install them into the
# project-local library (git-ignorable) with:
#
#   mkdir -p .Rlib
#   Rscript -e 'install.packages(c("BiocManager", "remotes"), lib = ".Rlib",
#               repos = "https://cloud.r-project.org")'
#   Rscript -e '.libPaths(c(".Rlib", .libPaths()));
#               BiocManager::install(c("Biostrings", "TFBSTools", "MatrixRider"),
#                                    lib = ".Rlib", update = FALSE, ask = FALSE);
#               remotes::install_github("matthiasheinig/tRap", lib = ".Rlib",
#                                       upgrade = "never")'
#
# Then, from the repository root:
#
#   Rscript scripts/make_reference_fixtures.R
#
# Writes tests/fixtures/matrixrider.csv and tests/fixtures/trap.csv, which
# tests/test_reference.py compares against seqaffinity (it skips if absent).
# Last run with R 4.5.1, Bioconductor 3.22, MatrixRider 1.42.0, TFBSTools 1.48,
# tRap 0.7 (github HEAD).
#
# Conventions of the two tools, read from their C sources (they differ from
# seqaffinity's; test_reference.py reproduces them from seqaffinity's window
# energies rather than expecting the default scores to agree):
#
#   * MatrixRider getSeqOccupancy(seq, pfm, cutoff = 0): lambda = 1. Zero counts
#     are REPLACED by 1 (non-zero counts are untouched), then LO = ln(f / bg)
#     with the PFMatrix bg (uniform here). Each window contributes
#     max(exp(LO+), exp(LO-)), the better strand only, not their sum. A window
#     containing N contributes 0. Counts must be integers.
#   * tRap affinity(pwm, seq): pwm is 4 x w (rows A, C, G, T). +1 pseudocount
#     per cell, lambda = 0.7, gc.content = 0.5 (uniform), Rmax =
#     exp(0.584 w - 5.66) uncapped. Per window the two strands are combined as
#     P+ + (1 - P+) P- (probability that at least one strand is bound), not
#     P+ + P-. N is scored as A and only a window whose LAST base is N gets 0
#     (illegalBase is reset per motif column in trap.cpp), so windows containing
#     N are excluded below from the *_valid columns.

local_lib <- file.path(getwd(), ".Rlib")
if (dir.exists(local_lib)) .libPaths(c(local_lib, .libPaths()))

suppressPackageStartupMessages({
  library(Biostrings)
})

fixture_dir <- file.path("tests", "fixtures")
seqs <- readDNAStringSet(file.path(fixture_dir, "ten.fa"))
seq_ids <- sub(" .*", "", names(seqs))

# MA0004.1 (Arnt), rows A, C, G, T; must match tests/fixtures/MA0004.1.jaspar.
arnt <- rbind(
  A = c(4L, 19L, 0L, 0L, 0L, 0L),
  C = c(16L, 0L, 20L, 0L, 0L, 0L),
  G = c(0L, 1L, 0L, 20L, 0L, 20L),
  T = c(0L, 0L, 0L, 0L, 20L, 0L)
)
w <- ncol(arnt)

# Starts (1-based) of motif windows that contain no N.
valid_windows <- function(s) {
  chars <- strsplit(as.character(s), "")[[1]]
  n_win <- length(chars) - w + 1
  vapply(seq_len(n_win), function(i) !any(chars[i:(i + w - 1)] == "N"), logical(1))
}
n_win <- width(seqs) - w + 1
n_valid <- vapply(seq_along(seqs), function(i) sum(valid_windows(seqs[[i]])), integer(1))

## MatrixRider ----------------------------------------------------------------
if (requireNamespace("MatrixRider", quietly = TRUE) && requireNamespace("TFBSTools", quietly = TRUE)) {
  pfm <- TFBSTools::PFMatrix(ID = "MA0004.1", name = "Arnt", profileMatrix = arnt)
  mr <- vapply(seq_along(seqs),
               function(i) MatrixRider::getSeqOccupancy(seqs[[i]], pfm, 0), numeric(1))
  out <- data.frame(seq_id = seq_ids, total_affinity = mr, n_win = n_win,
                    n_valid = n_valid, lambda = 1,
                    pseudocount = "zero counts replaced by 1",
                    strands = "max", ambiguous_windows = "contribute 0",
                    version = as.character(packageVersion("MatrixRider")))
  write.csv(out, file.path(fixture_dir, "matrixrider.csv"), row.names = FALSE)
  message("wrote matrixrider.csv")
} else {
  message("MatrixRider/TFBSTools not installed; skipping")
}

## tRap -------------------------------------------------------------------------
if (requireNamespace("tRap", quietly = TRUE)) {
  lam <- 0.7
  rmax <- exp(0.584 * w - 5.66)
  aff <- function(s, ...) tRap::affinity(arnt, as.character(s), Rmax = rmax, lambda = lam, ...)
  rows <- lapply(seq_along(seqs), function(i) {
    s <- seqs[[i]]
    ok <- valid_windows(s)
    rc <- reverseComplement(s)
    native <- aff(s)                                   # both strands, OR-combined
    or_win <- aff(s, slide = TRUE)                     # per window, OR-combined
    plus_win <- aff(s, slide = TRUE, both.strands = FALSE)
    minus_win <- rev(aff(rc, slide = TRUE, both.strands = FALSE))  # forward starts
    data.frame(
      seq_id = seq_ids[i],
      trap_native = native,
      trap_or_valid = sum(or_win[ok]),
      trap_sum_valid = sum(plus_win[ok]) + sum(minus_win[ok]),
      n_win = length(ok),
      n_valid = sum(ok)
    )
  })
  out <- do.call(rbind, rows)
  out$lambda <- lam
  out$pseudo_count <- 1
  out$gc_content <- 0.5
  out$Rmax <- rmax
  out$version <- as.character(packageVersion("tRap"))
  write.csv(out, file.path(fixture_dir, "trap.csv"), row.names = FALSE)
  message("wrote trap.csv")
} else {
  message("tRap not installed; skipping")
}
