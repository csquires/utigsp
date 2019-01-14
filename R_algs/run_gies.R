suppressMessages(library(pcalg))

args = commandArgs(trailingOnly=TRUE)
lambda = as.numeric(args[1])
sample_path = args[2]

obs_sample_path = paste(sample_path, '/observational.txt', sep='')
all_data = read.table(obs_sample_path)
all_settings = rep(1, nrow(all_data))
setting2targets = list()
setting2targets[[1]] = integer(0)

iv_sample_folder = paste(sample_path, '/interventional/', sep='')
i = 2
for (file in list.files(iv_sample_folder)) {
    iv_data = read.table(paste(iv_sample_folder, file, sep=''))
    known_iv_str = strsplit(strsplit(file, ';')[[1]][1], '=')[[1]][2]
    known_ivs = as.numeric(strsplit(known_iv_str, ',')[[1]])
    known_ivs = known_ivs + 1
    all_data = rbind(all_data, iv_data)
    all_settings = c(all_settings, rep(i, nrow(iv_data)))
    setting2targets[[i]] = known_ivs
    i = i + 1
}

gies_score_fn <- new("GaussL0penIntScore", all_data, setting2targets, all_settings, lambda=lambda*.05*log(nrow(all_data))) # BIC score
gies.fit <- gies(gies_score_fn)
weights = gies.fit$repr$weight.mat()

csv_name = paste(sample_path, '/estimates/gies/lambda=', formatC(lambda, format='e', digits=2),'.txt', sep='')
write.table(weights, file=csv_name, row.names=FALSE, col.names=FALSE)

