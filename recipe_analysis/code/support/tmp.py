# remove all digits and parenthesis content from ingredients
txt = txt.translate(self.remove_punctuation).translate(self.remove_digits)
txt = re.sub(r'\(*\)', '', txt)
print(txt)
sys.stdout.flush()

# check whether string contains nonsensical characters when removing all of those with regular expression
if re.findall('[^a-zA-Z\s]', self.substitution.sub('', txt)):
    to_del.append(i)

# do the stemming
fin = []
for word in txt.split():
    if len(word) > 0:
        fin.append(self.stemmer.stem(re.sub(r'[^\w\s]', '', word)))
fin = ' '.join(fin)

X.loc[i, col] = fin