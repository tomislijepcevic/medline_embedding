zcat citations.train.xml.gz | java -ss6000k -cp $CLASSPATH -Xmx1G -Xms1G \
    gov.nih.nlm.nls.mti.trainer.OVATrainer \
    gov.nih.nlm.nls.mti.textprocessors.MEDLINEXMLTextProcessor "" \
    gov.nih.nlm.nls.mti.featuresextractors.BinaryFeatureExtractor "-l -n -c -f1" \
    configuration.txt trie_new.gz classifiers_new.gz 2> out.log > out.txt
