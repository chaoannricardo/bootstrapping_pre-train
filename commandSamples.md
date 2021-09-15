# Original Dataset
```bash
# pretrain self-supervised
python pretrain_self.py --output_model_file ./models/210910_selfSupervised_test --data_dir ../KnowledgeGraph_materials/data_kg/bootstrapnet_data/boot_pretrain_data_revised/ --feature_type random --cpu

# pretrain supervised
python pretrain_sup.py --input_model_file ./models/210910_selfSupervised --output_model_file ./models/21091ˋ_supervised --data_dir ../KnowledgeGraph_materials/data_kg/bootstrapnet_data/boot_pretrain_data_revised/ --feature_type random --cpu

# fine tune
python fine_tune.py --input_model_file ./models/210910_supervised --output_model_file ./models/210914_fineTuned --dataset ../KnowledgeGraph_materials/data_kg/bootstrapnet_data/boot_pretrain_data_revised/CoNLL/ --feature_type random --cpu
```




# KG Dataset
```bash
python pretrain_self.py --output_model_file ./models/210911_KG --data_dir ../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_GBN/ --feature_type random


```

