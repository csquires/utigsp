suppressMessages(library('InvariantCausalPrediction'))
suppressMessages(library('nonlinearICP'))

# === PARSE COMMAND LINE ARGUMENTS
args = commandArgs(trailingOnly=TRUE)
alpha = as.numeric(args[1])
sample_path = args[2]
nonlinear = as.logical(args[3])

# === LOAD DATA INTO A MATRIX AND A VECTOR SPECIFYING THE EXPERIMENT INDEX
obs_sample_path = paste(sample_path, '/observational.txt', sep='')
iv_sample_folder = paste(sample_path, '/interventional/', sep='')
obs_data = data.matrix(read.table(obs_sample_path))

nnodes = ncol(obs_data)
amat = matrix(0, nnodes, nnodes)
# === FOR EACH NODE, FIND PARENTS
for (node in 1:ncol(obs_data)) {
    # === LOAD DATA FOR ALL INTERVENTIONS THAT ARE NOT ON THIS VARIABLE
    all_settings = rep(1, nrow(obs_data))
    i = 2
    all_data = obs_data
    for (file in list.files(iv_sample_folder)) {
        known_iv_str = strsplit(strsplit(file, ';')[[1]][1], '=')[[1]][2]
        known_ivs = as.numeric(strsplit(known_iv_str, ',')[[1]])
        known_ivs = known_ivs + 1
        if (!(node %in% known_ivs)) {
            iv_data = data.matrix(read.table(paste(iv_sample_folder, file, sep='')))
            all_data = rbind(all_data, iv_data)
            all_settings = c(all_settings, rep(i, nrow(iv_data)))
            i = i + 1
        }
    }
    # === ICP NEEDS AT LEAST TWO ENVIRONMENTS
    if (i > 2) {
        if (nonlinear) {
            res <- nonlinearICP(all_data[,-node], all_data[,node], as.factor(all_settings), alpha=alpha)
        } else {
            res <- ICP(all_data[,-node], all_data[,node], all_settings, alpha=alpha, showAcceptedSets=FALSE, showCompletion=FALSE)
        }
        print('==============================')
        print('node:')
        print(node)
        print(res)
        parent_ixs = Reduce(intersect, res$acceptedSets)
        # parent_ixs2 = which(res$pvalues < alpha)
        other_nodes = 1:ncol(all_data)
        other_nodes = other_nodes[-node]
        parents = other_nodes[parent_ixs]
        # parents2 = other_nodes[parent_ixs2]

        # print('parents equal:')
        # print(all.equal(parents, parents2))

        amat[parents, node] = 1
    }

}

print(amat)
csv_name = paste(sample_path, '/estimates/icp/alpha=', formatC(alpha, format='e', digits=2),'.txt', sep='')
write.table(amat, file=csv_name, row.names=FALSE, col.names=FALSE)


