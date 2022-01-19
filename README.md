# FinEvent
Code for "Reinforced, Incremental and Cross-lingual Event Detection From Social Messages" Accepted by T-PAMI.
This is an extension of The Web Conference 2021 paper [Knowledge-Preserving Incremental Social Event Detection via Heterogeneous GNNs](https://arxiv.org/pdf/2101.08747.pdf).

# Twitter Datasets
The Twitter dataset [1] is collected to evaluate social event detection methods. After filtering out repeated and irretrievable tweets, the dataset contains 68,841 manually labeled tweets related to 503 event classes, spread over a period of four weeks. To conduct the cross-lingual experiment, we additionally collect French Twitter dataset containing 64,516 labeled tweets related to 257 event classes and spread over about 3 weeks (a period of 23 days). Please find the original dataset at http://mir.dcs.gla.ac.uk/resources/

# Function Mode
online + offline + cross-lingual

To reduce the complicated data splitting process, we release the splitted datasets on [Google Drive](https://drive.google.com/file/d/1OCZ_OokyTA3mKe18dRSizhS0fJTcVfuD/view?usp=sharing)

## To run FinEvent Incremental
step 1. run utils/generate_initial_features.py to generate the initial features for the messages

step 2. run utils/custom_message_graph.py to construct incremental message graphs. To construct small message graphs for test purpose, set test=True when calling construct_incremental_dataset_0922(). To use all the messages (see Appendix of the paper for a statistic of the number of messages in the graphs), set test=False.

step 3. run utils/save_edge_index.py in advance to acclerate the training process.

step 4. run main.py

## To run FinEvent Offline
step 1-3 ditto (change the file path)

step 4. run offline.py

## To run FinEvent Cross-lingual
step 1-3 ditto (change the file path)

step 4. run main.py (Train a model from high-source dataset)

step 5. run resume.py

# Baselines
For Word2vec[3], we use the [spaCy pre-trained vectors](https://spacy.io/models/en#en_core_web_lg).

For [LDA](https://radimrehurek.com/gensim/models/ldamodel.html)[4], [WMD](https://tedboy.github.io/nlps/generated/generated/gensim.similarities.WmdSimilarity.html#gensim.similarities.WmdSimilarity)[5], [BERT](https://github.com/huggingface/transformers)[6], [PP-GCN](https://github.com/RingBDStack/PPGCN)[7] and [KPGNN](https://github.com/RingBDStack/KPGNN.git), we use the open-source implementations.

We implement EventX[8] with Python 3.7.3 and BiLSTM[9] with Pytorch 1.6.0. Please refer to the baselines folder. 

# Citation
If you find this repository helpful, please consider citing the following paper.

# Reference
[1] Andrew J McMinn, Yashar Moshfeghi, and Joemon M Jose. 2013. Building a large-scale corpus for evaluating event detection on twitter. In Proceedings of the CIKM.ACM, 409–418.

[2] Xiaozhi Wang, Ziqi Wang, Xu Han, Wangyi Jiang, Rong Han, Zhiyuan Liu, Juanzi Li, Peng Li, Yankai Lin, and Jie Zhou. 2020. MAVEN: A Massive General Domain
Event Detection Dataset. In Proceedings of EMNLP.

[3] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient estimation of word representations in vector space. In Proceedings of ICLR.

[4] David M Blei, Andrew Y Ng, and Michael I Jordan. 2003. Latent dirichlet allocation. JMLR 3, Jan (2003), 993–1022.

[5] Matt Kusner, Yu Sun, Nicholas Kolkin, and Kilian Weinberger. 2015. From word embeddings to document distances. In Proceedings of the ICML. 957–966.

[6] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).

[7] Hao Peng, Jianxin Li, Qiran Gong, Yangqiu Song, Yuanxing Ning, Kunfeng Lai, and Philip S. Yu. 2019. Fine-grained event categorization with heterogeneous graph convolutional networks. In Proceedings of the IJCAI. 3238–3245.

[8] Bang Liu, Fred X Han, Di Niu, Linglong Kong, Kunfeng Lai, and Yu Xu. 2020. Story Forest: Extracting Events and Telling Stories from Breaking News. TKDD 14, 3 (2020), 1–28.

[9] Alex Graves and Jürgen Schmidhuber. 2005. Framewise phoneme classification with bidirectional LSTM and other neural network architectures. Neural networks 18, 5-6 (2005), 602–610.