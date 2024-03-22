from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
import warnings
import os
from datetime import datetime as dt
import pandas as pd
from time import time
import glob

from torchkge.evaluation import *


def timer_func(func):
    # This function shows the execution time of the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

@timer_func
def evaluate_emb_model(emb_model, kg_eval, task, device, logger):
    """
    Evaluate the trained embedding model on a knowledge graph.

    Parameters
    ----------
    emb_model : torchkge.models.xxx
        The embedding model to be evaluated.
    kg_eval : torchkge.data_structures.KnowledgeGraph
        The knowledge graph used for evaluation.

    Returns
    -------
    None
    """
        
    logger.info(f'{dt.now()} - Evaluating..')
    b_size = 264 # Lower batch size if OOM error during evaluation

    match task:
        case 'link-prediction':
            evaluator = LinkPredictionEvaluator(emb_model, kg_eval)
            evaluator.evaluate(b_size=b_size, verbose=True)
            
        case 'relation-prediction':
            evaluator = RelationPredictionEvaluator(emb_model, kg_eval)

            use_cuda = next(evaluator.model.parameters()).is_cuda

            if use_cuda:
                dataloader = DataLoader(evaluator.kg, batch_size=b_size, use_cuda='batch')
                evaluator.rank_true_rels = evaluator.rank_true_rels.cuda()
                evaluator.filt_rank_true_rels = evaluator.filt_rank_true_rels.cuda()
            else:
                dataloader = DataLoader(evaluator.kg, batch_size=b_size)

            all_scores, all_true_ranks = torch.tensor([]).to(device), torch.tensor([]).to(device)
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                                    unit='batch', disable=(not True),
                                    desc='Relation prediction evaluation'):
                h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]
                h_emb, t_emb, r_emb, candidates = evaluator.model.inference_prepare_candidates(h_idx, t_idx, r_idx, entities=False)

                scores = evaluator.model.inference_scoring_function(h_emb, t_emb, candidates)
                filt_scores = filter_scores(scores, evaluator.kg.dict_of_rels, h_idx, t_idx, r_idx)

                if not evaluator.directed:
                    scores_bis = evaluator.model.inference_scoring_function(t_emb, h_emb, candidates)
                    filt_scores_bis = filter_scores(scores_bis, evaluator.kg.dict_of_rels, h_idx, t_idx, r_idx)

                    scores = cat((scores, scores_bis), dim=1)
                    filt_scores = cat((filt_scores, filt_scores_bis), dim=1)

                true_data = scores.gather(1, r_idx.long().view(-1, 1))
                
                ranks = (scores >= true_data).sum(dim=1).detach()
                evaluator.rank_true_rels[i * b_size: (i + 1) * b_size] = ranks
                evaluator.filt_rank_true_rels[i * b_size: (i + 1) * b_size] = (filt_scores >= true_data).sum(dim=1).detach()

                ranks = (scores >= true_data).sum(dim=1).detach()
                max_scores = torch.argmax(scores, dim=1)
                all_scores = torch.cat((all_scores, max_scores), dim=0)
                all_true_ranks = torch.cat((all_true_ranks, ranks), dim=0)

            evaluator.evaluated = True

            if use_cuda:
                evaluator.rank_true_rels = evaluator.rank_true_rels.cpu()
                evaluator.filt_rank_true_rels = evaluator.filt_rank_true_rels.cpu()


            # Compute the confusion matrix for the relation prediction task
            logger.info(kg_eval.rel2ix)
            # Convert tensors to numpy arrays
            all_scores_np = all_scores.cpu().numpy()
            all_true_ranks_np = all_true_ranks.cpu().numpy()
            cm = confusion_matrix(all_true_ranks_np, all_scores_np)

            # logger.info the confusion matrix
            logger.info(f'\n {cm}')
        case _:
            raise ValueError(f'Unknown task {task}')

    # Log results to logfile
    logger.info(f'{dt.now()} - EMBEDDING MODEL EVALUATION RESULTS:')
    logger.info(f'Task : {task}')
    for k in [1, 3, 5, 10]:
        logger.info(f'Hit@{k} : {evaluator.hit_at_k(k)[0]}')
    logger.info(f'Mean Rank : {evaluator.mean_rank()[0]}')
    logger.info(f'MRR : {evaluator.mrr()[0]}')

    # Log results to wandb
    # for k in range(1, 11):
    #     wandb.log({f'Hit@{k}': evaluator.hit_at_k(k)[0]})
    # wandb.log({'Mean Rank': evaluator.mean_rank()[0]})
    # wandb.log({'MRR': evaluator.mrr()[0]})

@timer_func
def load_celegans(keywords, sep):
    print(f'{dt.now()} - Querying celegans SPARQL endpoint with the following queries : {keywords}.')
    queries = queries_from_features(keywords)
    query_db(queries, sep)
    return 'query_result.txt'

def load_by_query(query):
    query = add_prefixes(query)
    query_db([query])

def query_db(queries, sep):
    """Queries the database with a SPARQL query that returns a graph (ie uses a CONSTRUCT clause)."""
    # Set up the SPARQL endpoint
    sparql = SPARQLWrapper("http://cedre-14a.med.univ-rennes1.fr:3030/WS287-rdf-coexp0.8/sparql") #TODO: make this endpoint default, but allow other endpoint in arguments
    warnings.filterwarnings("ignore")
    for query in tqdm(queries, desc="Querying SPARQL endpoint..."):

        # Set the query
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        # Execute the query and parse the response
        results = sparql.query()
        results = results.convert()
        # Store the constructed subgraph
        try:
            with open('query_result.txt', 'a') as f:
                for s, p, o in results:
                    if any(x.toPython() == '' for x in (s, p, o)): # Checks if the triple is complete
                        continue
                    f.write(f'{s}{sep}{p}{sep}{o}\n')       
        except:
            raise Exception("Check that the query output is a triple like ?s ?p ?o. Reminder: You must use a CONSTRUCT query")
    print(f'{dt.now()} - Query executed !')

def queries_from_features(keywords):
    file_paths = glob.glob("sparql_queries/*")
    file_names = [os.path.basename(file_path).replace('.txt', '') for file_path in file_paths]
    prefixes = open("sparql_queries/PREFIXES.txt", "r").read()

    ret = []
    for keyword in keywords:
        if keyword not in file_names:
            raise Exception("Unknown keyword: " + keyword)
        query = open("sparql_queries/" + keyword + ".txt", "r").read()
        ret.append(f"{prefixes}\n{query}")
    return ret


# Legacy code to load data in the format required by PhenoGeneRanker. Implementation incomplete.
# def load_pgr(keywords, sep):
#     feature_type = {
#         "pgr-gene-pheno" : "bipartite",
#         "pgr-gene-gene" : "bipartite",
#         "pgr-gene-disease" : "bipartite",
#         "pgr-pheno-pheno" : "bipartite",
#         "pgr-disease-disease" : "bipartite",
#     }

#     for keyword in keywords:
#         # Retrieve data files
#         load_celegans([keyword], sep)
#         with open('query_result.txt', 'r') as f:
#             lines = f.readlines()
#         with open(f'{keyword}.txt', 'a') as f:
#             f.write(f'from{sep}to{sep}weight\n')
#             for line in lines:
#                 parsed = line.split(sep)
#                 parsed = parsed[0] + sep + parsed[2].strip('\n') + sep + '1.000\n'
#                 f.write(parsed)
#         os.remove('query_result.txt')

#     # Create input file
#     with open('input.txt', 'a') as f:
#         f.write(f'type{sep}file_name\n')
#         for keyword in keywords:
#             f.write(f'{feature_type[keyword]}{sep}{keyword}.txt\n')
#     return 'input.txt'

if __name__ == "__main__":
    # Change directory to the current file path
    current_file_path = os.path.realpath(__file__)
    current_dir = os.path.dirname(current_file_path)
    os.chdir(current_dir)

    if os.path.exists("query_result.txt") == True:
        os.remove("query_result.txt")

    keywords = ['molecular_entity', 'phenotype', 'interaction', 'disease_plus_ortho',
     'disease-ontology', 'phenotype-ontology', 'expression_pattern',
      'lifestage-ontology'] # , 'go-ontology', 'go-annotation'
    load_celegans(keywords, sep=' ')
    
    print("All queries executed, results saved in query_result.txt")
