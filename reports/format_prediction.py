"""
Converts output of predict.py into human-friendly format
"""
from SPARQLWrapper import SPARQLWrapper, CSV
import argparse


prefix = 'PREFIX wbgene: <https://wormbase.org/species/c_elegans/gene/WBGene> PREFIX wbpheno: <https://wormbase.org/species/all/phenotype/WBPhenotype:> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> '

def transform_iri(iri):
    """transforms long iri to short iri"""
    if 'WBGene' in iri:
        return 'wbgene:{}'.format(iri[-8:])
    elif 'WBPhenotype' in iri:
        return 'wbpheno:{}'.format(iri[-7:])

def get_label(iri, endpoint):
    """Queries the database with a SPARQL query that returns the iri's label (follows rdfs:label relation)."""

    #shorten iri
    short_iri = transform_iri(iri)
    # Set up the SPARQL endpoint
    sparql = SPARQLWrapper(endpoint)
    query = '{} SELECT ?label WHERE {{ {} rdfs:label ?label . }}'.format(prefix, short_iri)

    # Set the query
    sparql.setQuery(query)
    sparql.setReturnFormat(CSV)

    # Execute the query and parse the response
    results = sparql.query()
    results = str(results.convert())
    results = results[11:-5]
    return results

#get_label('https://wormbase.org/species/c_elegans/gene/WBGene00006993')
#get_label('https://wormbase.org/species/all/phenotype/WBPhenotype:0000643')

def humanify_table(file, endpoint):

    md_file = file + '.md'
    csv_file = file + '.csv'
    with open(file, 'r') as rf:
        with open(md_file, 'w') as of:
            with open(csv_file, 'w') as cf:
                for line in rf:
                    if line.startswith('input'):
                        header = line.split(',')
                        of.write('| {} | {} | {} | {} | {} |\n'.format(header[0], header[1], header[2], header[3], header[4].strip()))
                        of.write('| :-- | :-- | :--: | :--: | :--: |\n')
                        cf.write('{},{},{},{},{}\n'.format(header[0], header[1], header[2], header[3], header[4].strip()))
                    else:
                        data = line.split(',')
                        infos = []
                        for element in data:
                            if element.startswith('https'):
                                infos.append(get_label(element, endpoint))
                            else:
                                infos.append(element)
                        of.write('| {} | {} | {} | {} | {} |\n'.format(infos[0], infos[1], infos[2], infos[3], infos[4].strip()))
                        cf.write('{},{},{},{},{}\n'.format(infos[0], infos[1], infos[2], infos[3], infos[4].strip()))

def get_ontology_leafs(nb_ancestors, ontology_name, endpoint):
    """Queries the database with a SPARQL query that returns the ontology terms that have at least nb_ancestors ancestors."""
    if ontology_name == "phenotype":
        key = 'heno'
    # Set up the SPARQL endpoint
    sparql = SPARQLWrapper(endpoint)
    query = 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> SELECT DISTINCT ?leaf1 WHERE {{ ?leaf1 rdfs:subClassOf{{{},}} ?leaf2 . FILTER(CONTAINS(str(?leaf1), "{}")) }}'.format(nb_ancestors,key)

    # Set the query
    sparql.setQuery(query)
    sparql.setReturnFormat(CSV)

    # Execute the query and parse the response
    results = sparql.query()
    results = str(results.convert())
    results = results[11:].split('\\r\\n')
    return results

def filter_ontology_leafs(file, nb_ancestors, ontology_name, endpoint):
    """
    Creates a new file containing only the lines which contain a phenotype in the leaf list (== that as a minimum of ancestors)
    """
    leaf_list = get_ontology_leafs(nb_ancestors, ontology_name, endpoint)
    output_file = '{}_filtered_{}_ancestors.txt'.format(file, nb_ancestors)
    with open(file, 'r') as f:
        with open(output_file, 'w') as of:
            for line in f:
                tested_phenotype = line.split(',')[1]
                if tested_phenotype in leaf_list:
                    of.writelines(line)

def get_domain_leafs(ancestor_name, endpoint):
    """Queries the database with a SPARQL query that returns the ontology terms that have at least nb_ancestors ancestors."""
    # if ontology_name == "phenotype":
    #     key = 'heno'
    # Set up the SPARQL endpoint
    sparql = SPARQLWrapper(endpoint)
    query = 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> SELECT DISTINCT ?leaf1 WHERE {{ ?leaf1 rdfs:subClassOf* ?leaf2 . ?leaf2 rdfs:label "{}" . }}'.format(ancestor_name)

    # Set the query
    sparql.setQuery(query)
    sparql.setReturnFormat(CSV)

    # Execute the query and parse the response
    results = sparql.query()
    results = str(results.convert())
    results = results[11:].split('\\r\\n')
    #print(results)
    return results

def filter_domain_leafs(file, domains_ancestors, endpoint):
    """
    Creates a new file containing only the lines which contain a phenotype in a domain of the ontology (== is a decendant of a term)
    """
    domains_list = []
    for domain in domains_ancestors:
        domains_list.append(get_domain_leafs(domain, endpoint))
    output_file = '{}_filtered_{}_domain.txt'.format(file, domains_ancestors)
    with open(file, 'r') as f:
        with open(output_file, 'w') as of:
            for line in f:
                tested_phenotype = line.split(',')[1]
                if tested_phenotype in domains_list[0] or tested_phenotype in domains_list[1]:
                    of.writelines(line)

def generate_phenotype_to_gene_prediction_query(file, cutoff=50):
    """
    Retrieves the predicted phenotypes from a file and generates a query file to use as input for predict.py
    """
    query_file = file[:-4] + '.rq'
    line_count = 0
    with open(file, 'r') as f:
        with open(query_file, 'w') as qf:
            while line_count < cutoff:
                line = f.readline()
                pheno = line.split(',')[1]
                qf.write('?,http://semanticscience.org/resource/SIO_001279,{}\n'.format(pheno))
                line_count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, type=str, help = 'Prediction file to convert.')
    parser.add_argument('-e', '--endpoint', required=False, default = "http://cedre-14a.med.univ-rennes1.fr:3030/WS287-rdf-coexp0.8/sparql", help = 'SPARQL endpoint url.')
    parser.add_argument('-a', '--nb_ancestors', required=False, default=False, help = 'If specified, will keep predicted terms that have at least nb_ancestors is the corresponding ontology. Else will filter by ontology domain/branch')
    parser.add_argument('-o', '--ontology', default = 'phenotype', help='Will use this ontology only to filter terms by nb_ancestors.')
    parser.add_argument('-q', '--generate_query', action = 'store_true', help = 'Will generate a file with queries to go from predicted phenotypes to make new genes predictions.')
    parser.add_argument('-c', '--cutoff', required=False, type=int, default=50, help = 'Used to limit the number of predicted phenotypes to include in the new query file for predictions.')
    parser.add_argument('-d', '--domains', required=False, default=("cell physiology phenotype", "cytoskeleton organization biogenesis variant"), help = 'Ontology terms used to filter predicted terms. Will keep only this term or its descendants')
    args = parser.parse_args()
    prediction_file= args.file
    endpoint = args.endpoint
    nb_ancestors = args.nb_ancestors
    ontology = args.ontology
    generate_query = args.generate_query
    cutoff = args.cutoff
    domains_ancestors = args.domains

    if not nb_ancestors: # if no number is provided for nb_ancestors program will run a filter on the ontology domains
        domains_ancestors = list(domains_ancestors)
        filter_domain_leafs(prediction_file, domains_ancestors, endpoint)
        filtered_file = '{}_filtered_{}_domain.txt'.format(prediction_file, domains_ancestors)
    else: # runs filter on number of ancestors (not on ontologydomain/branch)
        filtered_file = '{}_filtered_{}_ancestors.txt'.format(prediction_file, nb_ancestors)
        filter_ontology_leafs(prediction_file, nb_ancestors, ontology, endpoint)
    
    if generate_query: #generates a query for finding genes associated to phenotypes
        generate_phenotype_to_gene_prediction_query(filtered_file, cutoff)
    
    humanify_table(filtered_file, endpoint) #provides labels instead of urls in the prediction file
