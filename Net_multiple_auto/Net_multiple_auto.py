
import numpy as np
import argparse
import json
import cv2
import os
import random

from utils              import u_getPath, u_listFileAll, u_saveList2File, u_fileList2array, u_mkdir, u_listDirs, u_fileNumberList2array, u_saveArray2File
from NetMultiple        import Autoencoder, plotImages
from sklearn.model_selection import train_test_split

from ClusteringModels   import clusteringKmeans, clusteringGMM, clusteringAffinityProp, clusteringSpectral, clusteringDBScan, clusteringSOM, clusteringBirch, defineNumberOfClusters, getGMMParameters, filterTrks

################################################################################
################################################################################
def getNoisy(img_path, norm_factor = 255, token = '', noise = .5):
    
    if os.path.isfile( img_path ):
        files = u_fileList2array(img_path)
    else:
        files = u_listFileAll(img_path, token)
    
    clean = []
    
    for file in files:
        print('Reading file: ', file)
        if os.name == 'posix':
            file = file.replace('y:', '/rensso/qnap')
        img         = cv2.imread(file, 0)/norm_factor # normalizing [0 1]
        clean.append(img)
    
    #get dimensions
    shape = clean[0].shape
    shape += (1,)

    #reshaping to net
    clean   = np.reshape(clean, (len(clean), shape[0] , shape[1], shape[2] )) 

    #adding noise to clean images
    noisy   = clean + noise * np.random.normal(loc=0.0, scale=1.0, size=clean.shape) 
    
    #clipping 0 1 
    noisy   = np.clip(noisy, 0., 1.)

    return clean, noisy, shape, files;

################################################################################
################################################################################
def noisyData(img_path, 
              token         = '', 
              noise         = 0.5, 
              norm_factor   = 255,
              directory     = './', 
              name          = '', 
              test_size     = 0.3):

    clean, noisy, shape, files = getNoisy(img_path, norm_factor, token, noise)

    # dividing in test and train
    x_train, x_test, y_train, y_test, idx, idy = train_test_split(
        clean, noisy, files, test_size=test_size, random_state=1)

    u_saveList2File (directory + '/' + name +'_tr.lst', idx)
    u_saveList2File (directory + '/' + name +'_ts.lst', idy)

    return x_train, x_test, y_train, y_test, shape;

################################################################################
################################################################################
def train(general, data):
    #noise factor that will be added to images
    noise       = general['noise']
    model_info  = general['models'][data['model_pos']]
    directory   = general['directory']

    u_mkdir(directory)
    
    name        = model_info['name']
    model_name  = directory + '/' + name + '.h5'
    flist       = model_info['flist']
    nfactor     = model_info['norm_factor']
    
    img_token   = data['img_token']
    bach_size   = data['batch_size']
    epochs      = data['epochs']
    test_size   = data['test_size']
    
    #x_train, x_test, x_train_noisy, x_test_noisy, shape = getNoisyDataMnist(noise)
    x_train, x_test, x_train_noisy, x_test_noisy, shape = noisyData(
        flist, img_token, noise, nfactor, directory, name, test_size)

    #plot_chart([x_train[0].reshape(128,128), x_train_noisy[0].reshape(128,128)], 1, 2)

    autoencoder = Autoencoder(shape, data['model_pos'])
    autoencoder.train(X_train   = x_train_noisy, 
                      X_train_  = x_train,
                      epochs    = epochs,
                      batch_size= bach_size,
                      X_test    = x_test_noisy, 
                      X_test_   = x_test,
                      out       = model_name,
                      dirname   = directory)
    
    print('Saving model in: ', model_name)


################################################################################
################################################################################
def test(general, data):
    #noise factor that will be added to images
    noise       = general['noise']
    model_info  = general['models'][data['model_pos']]
    directory   = general['directory']
    
    nfactor     = model_info['norm_factor']
    name        = model_info['name']
    model_name  = directory + '/' + name + '.h5'
    flist       = directory + '/' + name + '_ts.lst'

    ntest       = data['ntest']
    out_file    = data['out_file']

    x_test, x_test_noisy, shape, files = getNoisy(flist, 
                                                  norm_factor   = nfactor, 
                                                  noise         = noise)

    autoencoder = Autoencoder(shape, data['model_pos'])
    
    autoencoder.evaluate(x_test_noisy, x_test, model_name, ntest, files, out_file)

################################################################################
################################################################################
def trainModels(general, data):
    noise   = general['noise']
    models  = general['models']

    for i in range(len(models)):
        data['model_pos'] = i 
        train(general, data) 

################################################################################
################################################################################
def featureExtraction(general, data):
    #noise factor that will be added to images
    noise       = general['noise']
    model_info  = general['models'][data['model_pos']]
    directory   = general['directory']
    #centersfile = general['centersfile']

    nfactor     = model_info['norm_factor']
    name        = model_info['name']
    model_name  = directory + '/' + name + '.h5'
    flist       = data['flist']
    base_name   = os.path.basename(flist).split('.')[0]
    out_name    = directory + '/' + base_name + '.ft'
    

    x_test, x_test_noisy, shape, files = getNoisy(flist, 
                                                  norm_factor   = nfactor, 
                                                  noise         = noise)

    autoencoder = Autoencoder(shape, data['model_pos'])
    
    features    = autoencoder.saveFeats(x_test_noisy, x_test, model_name)

    #features    = addcentersFeat(features, centersfile)

    print('Save features in: ', out_name)
    np.savetxt(out_name, features,
               header   = flist )

################################################################################
################################################################################
def saveFeats(general, data):
    noise   = general['noise']
    models  = general['models']
    
    for i in range(len(models)):
        data['model_pos']   = i 
        model_info          = general['models'][data['model_pos']]
        data['flist']       = model_info['flist'] 
        featureExtraction(general, data) 

################################################################################
################################################################################
def joinFeats():
    pass

################################################################################
################################################################################
# adds to end of each line the number of centers of each tracklet
def addcentersFeat(features, centersfile):
    
    centers  = u_fileNumberList2array(centersfile)
    new_feat = []

    for i in range(len(features)):
        new_feat.append(  np.append( features[i] , centers[i] ) )

    return new_feat

#...............................................................................
def clustering(general, data):
    
    model_info  = general['models']
    directory   = general['directory']
    methods     = data['methods']
    args        = data['args']

    aflist      = []
    for model in model_info:
        aflist.append(model['name'])

    # loading features .........................................................
    afeatures   = []
    for flist in aflist:
        afeatures.append( np.loadtxt (directory + '/' + flist + '.ft') )

    #includinng center number to end of features................................
    #numcfile    = general['numcfile']     
    #numc        = u_fileNumberList2array(numcfile)
    #numc        = np.array(numc).reshape(len(numc), 1)
    #feats       = np.concatenate ((afeatures[0], afeatures[1], numc), axis =1 )
    
    #normal pipeline............................................................
    feats    = np.concatenate ((afeatures[0], afeatures[1]), axis =1 )

    # loading labels............................................................ 
    labels  = u_fileList2array(model_info[0]['flist']) 

    # clustering 
    '''
    Affinity does not need the number of clusters
    DBSCAN has fixed eps 
    SOM has fixed neurons
    '''
    funcs   = { "KMeans"    : clusteringKmeans,
                "GMM"       : clusteringGMM,
                "Affinity"  : clusteringAffinityProp,
                "Spectral"  : clusteringSpectral,
                "DBScan"    : clusteringDBScan,
                "SOM"       : clusteringSOM,
                "Birch"     : clusteringBirch
              } 

    for method in methods:
        funcs[method](feats, labels, args, directory)

################################################################################
################################################################################
def visualizeClusters(general, data):
    directory       = general['directory']
    dir_name        = data['sub_dir']
    nsamples        = data['nsamples']
    width           = data['width']
    height          = data['height']
    img_size        = data['img_size']
    nsamples        = data['nsamples']      # number of samples to show
    m_n_clusters    = data['max']           # number of maximum clusters to show
    reverse_flag    = data['reverse']       # 1 increase sort 0 decrease sort
    save_flag       = data['savefig']       # 1 save figs to png 

    #...........................................................................
    xfactor = width/img_size
    yfactor = height/img_size

    for sub_directory, dirs, files in os.walk(directory): 
        if sub_directory.find(dir_name) >= 0:
            if os.path.isfile(sub_directory + '/data.json'):
                info    = json.load(open ( sub_directory + '/data.json' ))
                cluster_files   = info['cluster_list_file']
                cluster_lens    = info['cluster_lens']

                seq = sorted(range(len(cluster_lens)), key=lambda k: cluster_lens[k])
                if reverse_flag:
                    seq = seq[::-1]

                cluster_files = u_fileList2array(cluster_files)

                if save_flag:
                    sub_dir_nm  = os.path.basename(sub_directory)
                    out_dir_figs  = sub_directory + '/'+ sub_dir_nm 
                    u_mkdir(out_dir_figs)

                #...........................................................................
                for pos in range( min( len(seq), m_n_clusters) ):
                    
                    flist   = cluster_files[seq[pos]] 
                    random.seed()
                    clist   = u_fileList2array(flist)
                    cdlist  = random.sample( list( range( len( clist ) ) ) , 
                                            min(nsamples, len(clist) ))
                    imgs        = []
                    flist_cd    = []
                    for cd in cdlist:
                        ifile   = clist[cd].replace('mtA.png', 'd3i.png') 
                        flist_cd.append(os.path.dirname(ifile))

                        tfile   = clist[cd].replace('_mtA.png', '.trk') 
                        f       = open(tfile)
                        point   = f.readline().split(',')[1]
                        x , y   = point.split(' ')
                        
                        img     = cv2.imread(ifile, cv2.COLOR_BGR2RGB)
                        cv2.circle(img, 
                                   (int(float(x)/xfactor), int(float(y)/yfactor)), 
                                   radius = 6, 
                                   color=(0,255,0), 
                                   thickness=3, 
                                   lineType=8, 
                                   shift=0)
                        imgs.append(img)

                    if save_flag == 0:
                        plotImages(imgs, 'Number of items: '+ str(cluster_lens[seq[pos]]))
                    else:
                        
                        base        = out_dir_figs + '/' + sub_dir_nm + '_cluster_' + '%03d_' % (pos)  + str(seq[pos])
                        ffig_name   = base + '.png'
                        plotImages(imgs, 'Number of items: '+ str(cluster_lens[seq[pos]]),
                                   ffig_name)
                        fcdlist     = base + '.lst'
                        u_saveList2File(fcdlist, flist_cd)
################################################################################
################################################################################
def splittingCLusters():
    pass

################################################################################
################################################################################
def preClustering(general, data):
    numcfile    = general['numcfile']
    stage       = data['stage']

    #...........................................................................
    #estimating clustering using gmm

    numc        = np.loadtxt(numcfile)
    numc        = numc.reshape(-1,1)
    
    #...........................................................................
    if stage == 0:

        defineNumberOfClusters(numc)
    #...........................................................................
    if stage == 1:
        directory   = general['directory']
        components  = data['components']

        means       = getGMMParameters(numc, components)
        
        out_file    = directory + '/means_.txt'
        np.savetxt(out_file, means)
        print('Means saved in:', out_file)
        print(means)

    #...........................................................................
    if stage == 10:
        labels  = u_fileList2array('y:/research/suported/trks/radial.lst')
        filterTrks(numc, labels) 
        

################################################################################
################################################################################    
################################ Main controler ################################
def _main():

    funcdict = {'train'             : train,
                'test'              : test,
                'train_models'      : trainModels,
                'save_feats'        : saveFeats,
                'feature_extraction': featureExtraction,
                'clustering'        : clustering,
                'visualize'         : visualizeClusters,
                'pre_clustering'    : preClustering}

    conf_f  = u_getPath('conf2.json')
    confs   = json.load(open(conf_f))

    #...........................................................................
    funcdict[confs['op_type']](confs['general'], confs[confs['op_type']])
    
   
################################################################################
################################################################################
############################### MAIN ###########################################
if __name__ == '__main__':
    _main()


################################################################################
################################################################################
#def getNoisyDataMnist(noise_factor):
#    (x_train, _), (x_test, _) = mnist.load_data()

#    x_train = x_train.astype('float32') / 255.
#    x_test = x_test.astype('float32') / 255.

#    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
#    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

#    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
#    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

#    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
#    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

#    shape = (28, 28, 1)
#    return [x_train, x_test, x_train_noisy, x_test_noisy, shape];