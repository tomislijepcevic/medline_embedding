cat predict.txt | java -cp $CLASSPATH \
    gov.nih.nlm.nls.mti.evaluator.Evaluator benchmark.test > benchmark.txt
