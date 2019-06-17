import cPickle
"""
this is the module that will be slightly different for me 
I will be using special bit block box that maps certain words into same bit block
"""
data_Name = "cotra"
vocab_file = "vocab_" + data_Name + ".pkl"

word, vocab = cPickle.load(open('save/' + vocab_file))
print (len(word))
input_file = "save/generator_sample.txt"
output_file = "speech/" + data_Name + "_" + input_file.split('_')[-1]
with open(output_file, 'w') as fout:
    with open(input_file) as fin:
        for line in fin:
            line = line.split()
            line = [int(x) for x in line]
            line = [word[x] for x in line]

            line = ' '.join(line) + '\n'
            fout.write(line)