suppressMessages(library('InvariantCausalPrediction'))

# === PARSE COMMAND LINE ARGUMENTS
args = commandArgs(trailingOnly=TRUE)
alpha = as.numeric(args[1])
sample_path = args[2]

# === LOAD DATA INTO A MATRIX AND A VECTOR SPECIFYING THE EXPERIMENT INDEX
obs_sample_path = paste(sample_path, '/observational.txt', sep='')
all_data = data.matrix(read.table(obs_sample_path))
all_settings = rep(1, nrow(all_data))
iv_sample_folder = paste(sample_path, '/interventional/', sep='')
i = 2
for (file in list.files(iv_sample_folder)) {
    iv_data = data.matrix(read.table(paste(iv_sample_folder, file, sep='')))
    all_data = rbind(all_data, iv_data)
    all_settings = c(all_settings, rep(i, nrow(iv_data)))
    i = i + 1
}

amat = matrix(0, 3, 3)
# === FOR EACH NODE, FIND PARENTS
for (node in 1:ncol(all_data)) {
    res <- ICP(all_data[,-node], all_data[,node], all_settings, alpha=alpha, showAcceptedSets=FALSE, showCompletion=FALSE)
    parent_ixs = Reduce(intersect, res$acceptedSets)
    other_nodes = 1:ncol(all_data)
    other_nodes = other_nodes[-node]
    parents = other_nodes[parent_ixs]
    amat[parents, node] = 1
}

csv_name = paste(sample_path, '/estimates/icp/alpha=', formatC(alpha, format='e', digits=2),'.txt', sep='')
write.table(amat, file=csv_name, row.names=FALSE, col.names=FALSE)


