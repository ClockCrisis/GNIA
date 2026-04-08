```python
├── cresci_2015/
│   ├── raw_data/
│   ├── processed_data/
│   ├── model/
│   ├── checkpoint/
│   ├── utils.py
│   ├── model.py
│   ├── preprocess_1.py
│   ├── preprocess_2.py
│   ├── preprocess_3.py
│   ├── train.py # train substitute/GCN/HGT/SimpleHGN/RGCN model
│   ├── Dataset.py
│   ├── preprocess.py # preprocess the dataset
│   ├── dataset_tool.py
│   ├── cat_decoder.py # train categorical property decoder
│   ├── num_decoder.py # train numerical property decoder
│   ├── gnia.py
│   ├── run_gnia.py # train attack model
│   ├── layer.py
│   ├── test_GCN.py # test the attack model on the GCN model
│   ├── test_HGT.py # test the attack model on the HGT model
│   ├── test_SimpleHGN.py # test the attack model on the SimpleHGN model
│   └── test_RGCN.py # test the attack model on the RGCN model
├── twibot_22/
│   ├── raw_data/
│   ├── processed_data/
│   ├── model/
│   ├── checkpoint/
│   ├── utils.py
│   ├── model.py
│   ├── preprocess_1.py
│   ├── preprocess_2.py
│   ├── preprocess_3.py
│   ├── train.py # train substitute/GCN/HGT/SimpleHGN/RGCN model
│   ├── Dataset.py
│   ├── preprocess.py # preprocess the dataset
│   ├── sub-graph.py
│   ├── dataset_spilt.py # select subgraph
│   ├── dataset_tool.py # divide the subgraph dataset
│   ├── cat_decoder.py # train categorical property decoder
│   ├── num_decoder.py # train numerical property decoder
│   ├── gnia.py
│   ├── run_gnia.py # train attack model
│   ├── layer.py
│   ├── test_GCN.py # test the attack model on the GCN model
│   ├── test_HGT.py # test the attack model on the HGT model
│   ├── test_SimpleHGN.py # test the attack model on the SimpleHGN model
│   └── test_RGCN.py # test the attack model on the RGCN model
└── readme.md
```



- **implement details**: 

   There are some changes in user numerical properties & user categorical properties due to the lack of relevant data
  
   1. numerical properties:
   
      - original: (dim=6)

         followers + followings + favorites + statuses + active_days + screen_name_length 

      - cresci-2015/twibot-22: (dim=5)
      
         followers + followings + statuses + active_days + screen_name_length
   
   2. categorical properties: 
   
      - original: (dim=11)

         protected + verified + default_profile_image + geo_enabled + contributors_enabled + is_translator + is_translation_enabled + profile_background_image + profile_user_background_image + has_extended_profile + default_profile

      - cresci-2015: (dim=1)

        default_profile_image

      - twibot-22: (dim=3)

         protected + verified + default_profile_image


#### How to reproduce:

1. specify the dataset by entering corresponding fold

   - cresci-15 : `cd cresci_15/`
   - twibot-22 : `cd twibot_22/`
   
2. preprocess the dataset by running

   `python preprocess.py`

3. train substitute/GCN/HGT/SimpleHGN/RGCN model by running

   `python train.py`

4. train numerical property decoder

   `python num_decoder.py`

5. train categorical property decoder

   `python cat_decoder.py`
   
6. for twibot-22 dataset, select and divide subgraphs

   `python dataset_spilt.py`
   `python dataset_tool.py`

7. train attack model

   `python run_gnia.py`

8. test the attack model

   `python test_GCN.py`
   `python test_HGT.py`
   `python test_SimpleHGN.py`
   `python test_RGCN.py`



Seeds used in the paper:

Cresci-2015: 904， 607， 827， 1208， 1005

TwiBot-22: 904， 1， 2， 1208， 1006