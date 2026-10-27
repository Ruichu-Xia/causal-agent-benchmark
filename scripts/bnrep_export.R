# Export Gaussian bnRep networks (bn.fit.gnet) as JSON: node, parents, coefficients, residual sd.
# Usage: Rscript scripts/bnrep_export.R <bnRep source>/data data/bnrep expenditure lexical algal2 liquefaction
args <- commandArgs(trailingOnly = TRUE)
src <- args[1]; out <- args[2]; names <- args[-(1:2)]
q <- function(s) paste0('"', s, '"')
for (nm in names) {
  e <- new.env(); load(file.path(src, paste0(nm, ".rda")), envir = e); fit <- get(ls(e)[1], envir = e)
  stopifnot(inherits(fit, "bn.fit.gnet"))
  nodes <- sapply(names(fit), function(v) {
    n <- fit[[v]]; co <- n$coefficients
    pars <- paste(sprintf('%s: %.17g', q(names(co)[-1]), co[-1]), collapse = ", ")
    sprintf('{"node": %s, "parents": {%s}, "sd": %.17g}', q(v), pars, n$sd)
  })
  writeLines(sprintf('{"name": %s, "nodes": [%s]}', q(nm), paste(nodes, collapse = ",\n")), file.path(out, paste0(nm, ".json")))
}
