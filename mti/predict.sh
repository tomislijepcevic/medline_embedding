zcat citations.test.xml.gz |java -ss6000k -cp $CLASSPATH \
    gov.nih.nlm.nls.mti.annotator.OVAAnnotator \
    gov.nih.nlm.nls.mti.textprocessors.MEDLINEXMLTextProcessor "" \
    gov.nih.nlm.nls.mti.featuresextractors.BinaryFeatureExtractor "-l -n -c" \
    trie_new.gz classifiers_new.gz > predict.txt 2> predict.log