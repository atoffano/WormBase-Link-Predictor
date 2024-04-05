#!/usr/bin/env python
import argparse
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def data_extractor(file:str) -> dict:
    """Extracts information from log file"""
    
    nodes = 0
    relation_types = 0
    triples = 0
    embedding_stats= list()
    train_loss = list()
    valid_loss = list()
    # start_time = datetime.date()
    # phase1_end = datetime.date()
    # phase2_end = datetime.date()

    pattern = " - Epoch "

    with open(file, 'r') as f:
        for line in f:
            if 'Start time:' in line:
                start_time = datetime.strptime(line.split('INFO:')[0].split(',')[0].strip(), '%Y-%m-%d %H:%M:%S')
            elif 'Training of Embedding Model done' in line:
                phase1_end = datetime.strptime(line.split('INFO:')[0].split(',')[0].strip(), '%Y-%m-%d %H:%M:%S')
            elif 'Classifier trained' in line:
                phase2_end = datetime.strptime(line.split('INFO:')[0].split(',')[0].strip(), '%Y-%m-%d %H:%M:%S')
            elif 'Number of entities' in line:
                entities = int(line.split(':')[-1].strip())
                if nodes == 0:
                    nodes = entities
                elif nodes != entities:
                    nodes = -1
            elif 'Number of relation types' in line:
                reltype = int(line.split(':')[-1].strip())
                if relation_types == 0:
                    relation_types = reltype
                elif relation_types != reltype:
                    relation_types = -1
            elif 'Number of triples' in line:
                triples += int(line.split(':')[-1].strip())
            elif 'method :' in line:
                method = line.split(':')[-1].strip()
            elif pattern in line:
                embedding_stats.append(line.strip())
            elif "Hit@1 :" in line:
                hit_at_1 = float(line.split(':')[-1].strip())
            elif "batch_size : " in line:
                batch_size = int(line.split(':')[-1].strip())
            elif 'n_epochs :' in line:
                epochs = int(line.split(':')[-1].strip())
            elif 'keywords :' in line:
                keys = line.split(':')[-1].strip()
                if keys != 'None':
                    keywords = line.split(' ')
                    if 'onto' in keywords:
                        ontologies = True
                    else:
                        ontologies = False
                    if 'direct' in keywords:
                        structure = 'Direct'
                    else:
                        structure = 'Indirect'
                else:
                    keywords = False
                    structure = 'Unknown'
                    ontologies = 'Unknown'
    
    training_length = phase2_end - start_time
    phase1_length = phase1_end - start_time
    phase2_length = phase2_end - phase1_end

    for element in embedding_stats:
        stats = element.split('|')[1]
        train_loss.append(float(stats.split(' ')[3].strip(',')))
        valid_loss.append(float(stats.split(' ')[6].strip()))
    
    best_epoch = valid_loss.index(min(valid_loss)) +1

    data = {'filename':file,
            'total_length': training_length,
            'embedding_length': phase1_length,
            'classifier_length': phase2_length,
            'nodes':nodes,
            'relation_types':relation_types,
            'triples':triples,
            'algorithm':method,
            'train_loss': train_loss,
            'valid_loss':valid_loss,
            'hit@1':hit_at_1,
            'batch_size':batch_size,
            'data':keywords,
            'ontologies': ontologies,
            'structure':structure,
            'epochs':epochs,
            'best_epoch':best_epoch,
            }
    
    return data

def figure(data) -> None:
    """
    Generates a train/validation graph and saves it as file.png
    """
    file = data['filename']
    hit1= data['hit@1']
    batch = data['batch_size']
    train_loss = data['train_loss']
    valid_loss = data['valid_loss']
    best_epoch = data['best_epoch']
    training_time = data['total_length']

    #train line
    xt = list(range(1, len(train_loss)+1, 1))
    yt = train_loss
    #validation line
    xv = xt
    yv = valid_loss
    fig, ax = plt.subplots()
    ax.plot(xt, yt, label = "training loss")
    ax.plot(xv, yv, label = "validation loss")
    #plot batch size
    plt.text(0.5, 0.8,'Batch_size = ' + str(batch), transform = ax.transAxes) #horizontalalignment='center', verticalalignment='center', transform = ax.transAxes
    #plot Hit @1 value
    plt.text(0.5, 0.75, 'Hit@1 = ' + str(hit1), transform = ax.transAxes)
    #plot line at best validation epoch
    plt.axvline(x = best_epoch, color = 'red')
    plt.text(0.5, 0.7, 'Best_epoch = ' + str(best_epoch), transform = ax.transAxes)
    #plt time taken for training
    plt.text(0.5, 0.65, 'Training time = ' + str(training_time), transform = ax.transAxes)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(file)
    plt.legend()
    plt.savefig(file + '.png')


def main():
    """
    Generates a graph of the training and evaluation loss by epoch if grapoh file does not exist.
    graph also includes batch size and Hit@1 score of the final model
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('file', help='log file to extract informations from')

    args = parser.parse_args()

    logfile = args.file
    if not os.path.isfile(logfile + '.png'):
        data = data_extractor(logfile)
        figure(data)

if __name__ == "__main__":
    main()

    ## TODO: IMPROVE DATA USAGE and INFOS DISPLAY ON GRAPH
    ## notably training length