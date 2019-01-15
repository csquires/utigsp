suppressMessages(library('InvariantCausalPrediction'))

# === PARSE COMMAND LINE ARGUMENTS
args = commandArgs(trailingOnly=TRUE)
alpha = args[1]
sample_path = args[2]

# === LOAD DATA INTO A MATRIX AND A VECTOR SPECIFYING THE EXPERIMENT INDEX
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

# === FOR EACH NODE, FIND PARENTS
for (node in 1:ncol(all_data)) {
    res <- ICP(all_data[,-node], all_data[,node], all_settings)
    print(res)
}
