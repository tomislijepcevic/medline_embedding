zcat citations.test.xml.gz | java -cp $CLASSPATH \
    gov.nih.nlm.nls.mti.evaluator.GetBenchmark \
    gov.nih.nlm.nls.mti.textprocessors.MEDLINEXMLTextProcessor "" > benchmark.test
