# Original Dataset
```bash
# pretrain self-supervised
python pretrain_self.py --output_model_file ./models/210910_selfSupervised_test --data_dir ../KnowledgeGraph_materials/data_kg/bootstrapnet_data/boot_pretrain_data_revised/ --feature_type random --cpu

# pretrain supervised
python pretrain_sup.py --input_model_file ./models/210919_selfSupervised --output_model_file ./models/210922_supervised --data_dir ../KnowledgeGraph_materials/data_kg/bootstrapnet_data/boot_pretrain_data_revised/ --feature_type random --cpu

# fine tune
python fine_tune.py --input_model_file ./models/210922_supervised --output_model_file ./models/210922_fineTuned --dataset ../KnowledgeGraph_materials/data_kg/bootstrapnet_data/boot_pretrain_data_revised/CoNLL/ --feature_type random --cpu
```




# KG Dataset
```bash
# pretrain self-supervised

## GBN Original dataset
python pretrain_self.py --output_model_file ./models/210923_KG_WorldChronology_selfSupervised --data_dir ../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_GBN_wholeNode_dependency/ --feature_type random --n_epoch 300 --cpu

## semiconductor
python pretrain_self.py --output_model_file ./models/210922_KG_Semiconductor_Filtered_selfSupervised --data_dir ../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_GBN_wholeNode_dependency/ --feature_type random --n_epoch 300 --cpu

## world chronology
python pretrain_self.py --output_model_file ./models/210926_KG_WorldChronology_Filtered_selfSupervised --data_dir ../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_GBN_wholeNode_dependency/ --feature_type random --n_epoch 300 --cpu

# pretrain supervised

## world chronology
python pretrain_sup.py --input_model_file ./models/210926_KG_WorldChronology_Filtered_selfSupervised --output_model_file ./models/210926_KG_WorldChronology_Filtered_Supervised --data_dir ../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_GBN_wholeNode_dependency/ --feature_type random --n_epoch 300

# fine tune

## GBN Original dataset
python fine_tune.py --input_model_file ./models/210922_KG_Semiconductor_Filtered_selfSupervised_encoder --output_model_file ./models/210922_KG_Semiconductor_Filtered_FineTuned --dataset ../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_GBN_wholeNode_dependency/doc_un_1/ --feature_type random --cpu

## world chronology
python fine_tune.py --input_model_file ./models/210926_KG_WorldChronology_Filtered_Supervised --output_model_file ./models/210926__WorldChronology_FineTuned --dataset ../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_GBN_wholeNode_dependency/doc_un_1/ --feature_type random --cpu


```

