#!/usr/bin/env python3
# Author Marcel H. Schubert
# date 28.06.2021
import numpy as np

##data loading
import ndjson
import jsonlines
import json

##system
import os
import sys
import argparse
import ast
import gc
import multiprocessing as mp
import psutil

##text processing
import re
import demoji
from textblob import TextBlob
import spacy

#date parsing
from dateutil.relativedelta import *
from dateutil.easter import *
from dateutil.rrule import *
from dateutil.parser import *
from datetime import *

##other stuff
import time

#parameters




##we use the results by Custodio et al. 2021 but the parameters may be amended
def get_linebytes(path, filename, test = False):
    linebytes = []
    i = 0
    # fileEnd = os.path.getsize(fname)
    filename = os.path.join(path, filename)
    with open(filename, 'r', encoding='utf-8') as f:

        nextLineByte = f.tell()
        while True:
            linebytes.append(nextLineByte)
            line = f.readline()
            nextLineByte = f.tell()  # returns the location of the next line

            if test:
                i += 1
                if i == 1:
                    break

            if not line or line == '':
                break
    if not test:
        with open(os.path.join(path, "linebytes.json"), 'w') as p:
            json.dump(linebytes, p)

    return linebytes

def chunkify(linebytes, chunksize):

    linebytes = [linebytes[i:(i+chunksize)] for i in range(0, len(linebytes), chunksize)]

    return linebytes


def check_done(linebytes, savepath, path, filename, ):
    #checks whether part of the job was already done before - enables sinterrupt and continue of tasks

    filename = os.path.join(path, filename)
    org = open(filename, 'r', encoding='utf-8')

    #use set implementation for comparison
    counter_dic = {}
    if os.path.exists(os.path.join(savepath, 'process', '_processed_ids.ndjson')):
        processed = set()
        with open(os.path.join(savepath, 'process' '_processed_ids.ndjson'), 'r+', encoding='utf-8') as f:
            for line in f:
                l = ndjson.loads(line)[0]
                counter_dic[l['byte']] = counter_dic.get(l['byte'], {})
                counter_dic[l['byte']][l['byte']] = counter_dic.get(l['byte'], {})


                counter_dic[l['byte']]['counter'] = counter_dic[l['byte']].get('counter', 0) +1
                counter_dic[l['byte']]['upper'] = counter_dic[l['byte']].get('upper', l['pprocessed'])
                if counter_dic[l['byte']]['counter'] == counter_dic[l['byte']]['upper']:
                    # make set from list of dic
                    processed.add(l['byte'])

    else:
        return linebytes
    linebytes = set(linebytes)
    linebytes = list(linebytes.difference(processed))

    return linebytes

def get_done(savepath, i):
    #checks whether a file of ids exists for which the tasks are already done
    if os.path.exists(os.path.join(savepath, 'process', '_processed_ids_part_{}.ndjson'.format(i))):
        done_ids = {}
        for j in [i-1, i, i+1]:
            if os.path.exists(os.path.join(savepath, 'process', '_processed_ids_part_{}.ndjson'.format(j))): #create latitude around i
                with open(os.path.join(savepath, 'process', '_processed_ids_part_{}.ndjson'.format(i)), 'r', encoding='utf-8') as f:
                    for line in f:
                        l = ndjson.loads(line)[0]
                        done_ids[l['tweetID']] = l['ID']
    else:
        done_ids = None

    return done_ids

def find_pol(text):
    #gets text sentiment and polarity using textblob
    return TextBlob(text).sentiment

def repl_special_characters(matcher):
    #deletes sppecial characters except when they belong to a list of predefined emoijis
    smile_space_re = re.compile(r"(:\\|:-/|:/|:-\\|:\)|;\)|:-\)|;-\)|:\(|:-\(|:-o|:o|<3|\s+)")
    ##keep part after contraction (not) as separate word
    if matcher.group(0) == "'":
        #return ' '
        return ''
    elif not re.search(smile_space_re, matcher.group(0)):
        return ''

    else:
        return matcher.group(0)

def make_ngrams_text(text,range_WORD = (1,2)):
    #makes word-based ngrams
    wordgrams = {key:[] for key in range(range_WORD[0], range_WORD[1]+1)}
    for i in range(len(text)-range_WORD[0]+1):
        for key in wordgrams.keys():
            if key+i < len(text)+1:
                wordgrams[key].append('-'.join(text[i:(i+key)]))

    return wordgrams

def make_ngrams_char(text, range_CHAR = (2,5)):
    #makes char-based ngrams
    chargrams = {key:[] for key in range(range_CHAR[0], range_CHAR[1]+1)}
    textlen = len(text)
    for i in range(textlen-range_CHAR[0]+1):
        for key in chargrams.keys():
            if key+i < textlen+1:
                chargrams[key].append(''.join(text[i:(i+key)]))


    return chargrams

def parse_spacy(ids, tweetIds, texts):
    #makes the spacy tokens (lemmas, pos, tag, dep etc.) and saves then in individual strings as tokens
    import srsly
    from spacy.symbols import ORTH
    #exchange patterns to assess contractions correctly
    patterns = srsly.read_json("../../spacy/ar_patterns.json")
    nlp = spacy.load('en_core_web_lg')
    old = nlp.remove_pipe("attribute_ruler")
    del old
    ar = nlp.add_pipe("attribute_ruler", before="lemmatizer")
    ar.add_patterns(patterns)

    #set inserted tags
    smileys = [":-o", ":o", "<3", r":\\", ":-/", ":/", r":-\\", ":)", ";)", ":-)", ";-)", ":(", ":-(", ]
    smile_re = re.compile(r"(:\\|:-/|:/|:-\\|:\)|;\)|:-\)|;-\)|:\(|:-\(|:-o|:o|<3)")
    tags = ["<URL>", "<HASHTAG>", "<USER>", "<EMOJI>", "<EMOTICON>", "<TIME>", "<NUMBER>", "<DATE>", "<BEG>", "<END>"]
    time_re = re.compile("[\d]+:[\d]+")
    www_re = re.compile(r'www\.[^\s]+')
    url_re = re.compile(r"http[s]*://[^\s]+")

    for el in smileys+tags:
        special_case = [{ORTH: el}]
        nlp.tokenizer.add_special_case(el, special_case)
    #for i in range(len(texts)):
    #    if not (re.search(url_re, texts[i]) or re.search(www_re, texts[i])):
    #        texts[i] = demoji.replace(texts[i], "<EMOJI>")
        #texts[i] = re.sub(smile_re, "<EMOJI>", texts[i])

    # make spacy model
    texts = nlp.pipe(texts, batch_size=500)
    processed = []
    for i, doc in enumerate(list(texts)):
        out = {'lemma': [], 'pos': [], 'tag': [], 'dep': []}
        out['tensor'] = doc.tensor
        out['id'] = ids[i]
        out['tid'] = tweetIds[i]
        num_tokens = doc.__len__()
        for j, token in enumerate(list(doc)):
            lemma = token.text
            lemma = re.sub(url_re, "<url>", lemma)
            lemma = re.sub(www_re, '<url>', lemma)
            lemma = re.sub(smile_re, "<EMOTICON>", lemma)

            #make sure our markers are placed correclty everywhere
            if j ==0:
                out['pos'].append('<beg>')
                out['tag'].append('<beg>')
                out['dep'].append('<beg>')

            elif j == num_tokens-1:
                out['pos'].append('<end>')
                out['tag'].append('<end>')
                out['dep'].append('<end>')

            elif token.pos_ != 'X':
                out['pos'].append(token.pos_)
                out['tag'].append(token.tag_)
                out['dep'].append(token.dep_)
            else:
                out['dep'].append(token.dep_)

            #check wheter we are talking about a date
            if re.findall("\d", lemma):
                if re.findall(time_re, lemma):
                    lemma = "<TIME>"
                else:
                    try:
                        res = parse(lemma)
                        lemma = "<DATE>"
                    except:
                        pass
                lemma = re.sub("[\d]+", "<NUMBERS>", lemma)

            if not (len(lemma) == 1 and re.findall(r"\W", lemma)):
                out['lemma'].append(lemma)

        processed.append(out)

    return processed

def parse_text_char(text, encase, asis=False):
    #parsing for character-based ngram tokens
    non_characters = re.compile(r"(?<!<URL|HTAG|USER|MOJI|ICON|TIME|MBER|OJI>|<END|<BEG)[^a-zA-Z0-9_\s]+(?![^a-zA-Z0-9_\s]*<EMOJI>|URL>|HASHTAG>|USER>|EMOJI>|EMOTICON>|TIME>|NUMBER>|END>|BEG>)")
    non_literal = re.compile(r"(?<!<URL|HTAG|USER|MOJI|ICON|TIME|MBER|OJI>|<END|<BEG)[<>]+(?![^a-zA-Z0-9_\s]*<EMOJI>|URL>|HASHTAG>|USER>|EMOJI>|EMOTICON>|TIME>|NUMBER>|END>|BEG>)")

    #for keeping every symbol in there and only doing characters
    if not asis:
        text = re.sub(non_characters, repl_special_characters, text)
    else:
        text = re.sub(non_literal, repl_special_characters, text)
    words = []
    chars = []
    word = ''
    token = ''
    startT = False
    for i in range(len(text)):
        char = text[i]

        ##for ease of reading later make white spaces to blanks
        if re.match(r"\s", char):
            char = '_'

        #if we have not a token-start or if we are not within a token
        if char != "<" and not startT:
            chars.append(char)
            #if the char is not a white space
            if char != '_':
                word = word + char
            #if whitespace append and set word to empty
            else:
                if word != '':
                    words.append(word.lower())
                    word = ''

        elif char == ">" or startT:
            #if we are on end of token we have full word and char
            token = token + char
            #if we have an emoji token of either kind and we not do encasing
            if not encase and re.fullmatch(r"(<EMOJI>|<EMOTICON>)", token):
                words.append(token.lower())
                chars.append(token)
                word = ''
                token = ''
                startT = False
            #if we are at end of token and the token is not an EMOJI/EMOTICON token
            elif char == ">" and re.search(r">", token) and not re.fullmatch(r"(<EMOJI>|<EMOTICON>)", token):
                words.append(token.lower())
                chars.append(token)
                word = ''
                token = ''
                startT =False

            #if we are at the end of encased emoji
            elif encase and re.fullmatch(r"(<EMOJI>|<EMOTICON>).+(<EMOJI>|<EMOTICON>)", token):
                words.append(token.lower())
                chars.append(token)
                word = ''
                token = ''
                startT = False


        elif char == "<" and not startT:
            startT = True
            token = token + char

    return ['<beg>'] + words + ['<end>'],  ['<BEG>'] + chars + ['<END>']

def parse_dist(t):
    # makes the text-distortion tokens

    #negative look-ahead and negative lookbehind
    characters = re.compile(r"(\w|(<EMOJI>.*?<EMOJI>)|(<EMOTICON>.*?<EMOTICON>)|<URL>|<HASHTAG>|<USER>|<EMOJI>|<EMOTICON>|<TIME>|NUMBER>)")
    space_collapse_re = re.compile("[\s]+")


    ##make textdistortion
    dist = re.sub(characters, '*', t)
    dist = re.sub(space_collapse_re, '_', dist)

    return dist

def parse_raw(id, tweetId, tweet, encase =[],
          spacy=False, filehandles=None):

    #makes general preprocessing steps for single tweet - less for the spacy part more for the char and word parts
    #print('in parse raw')
    #sys.stdout.flush()
    # regex for rpeprocessing
    www_re = re.compile('www\.[^\s]+')
    url_re = re.compile("http[s]*://[^\s]+")
    mention_re = re.compile("(^|[\W\s])@[a-zA-Z0-9_]+[\s]")
    smile_re = re.compile(r"(\s:\\|:-/|:/\s|:-\\|:\)|;\)|:-\)|;-\)|:\(|:-\(|:-o|:o|<3)")
    not_ascii_re = re.compile("([^\x00-\x7F]+)")
    time_re = re.compile("(^|\D)[\d]+:[\d]+")
    letter_numbers_re = re.compile("((^|\D)[\d]+[.'\d]*\D[^\s])")
    space_collapse_re = re.compile("[\s]+")
    hashtag_re = re.compile("#")

    tag_dic = {'url': "<URL>",
               'tag': " <HASHTAG> ",
               'mention': " <USER> ",
               'emoji': "<EMOJI>",
               'smile': "<EMOTICON>",
               'time': "<TIME>",
               'number': "<NUMBER>",
               'space' : "_"
               }
    #initial length
    num_char_b = len(tweet)

    t = tweet.replace("\n", " ")
    ##this must be done irrespective of what input is generated
    polarity, subjectivity = find_pol(tweet)

    if not spacy:
        f = re.findall(mention_re, t)
        num_mention = len(f)
    t = re.sub(mention_re, tag_dic['mention'], t)

    if spacy:
        t = re.sub("#", "", t)
    if not spacy:
        f = re.findall("#", t)
        num_tags = len(f)
        t = re.sub(hashtag_re, tag_dic['tag'], t)
    if not spacy:
        f = re.findall(letter_numbers_re, t)
        num_numbers = len(f)
    t = re.sub(letter_numbers_re, tag_dic['number'], t)

    if spacy:
        t = re.sub(space_collapse_re, ' ', t)
        t = demoji.replace_with_desc(t, sep='<EMOJI>')
        t = re.sub(not_ascii_re, '', t)
        return '<BEG> '+t+' <END>'

    f = re.findall(url_re, t)
    num_urls = len(f)
    t = re.sub(url_re, tag_dic['url'], t)

    f = re.findall(www_re, t)
    num_urls += len(f)
    t = re.sub(www_re, tag_dic['url'], t)

    f = re.findall(time_re, t)
    num_times = len(f)
    t = re.sub(time_re, tag_dic['time'], t)

    filehandles['polarity']['writer'].write({'ID': id, 'tweetID': tweetId, 'polarity':polarity, 'subjectivity':subjectivity})

    #check whether to encase emojis or whether to substitute them by a token
    if encase:
        for key in encase:
            if key != 'emoji':
                tag_dic[key] = tag_dic[key]+'\g<0>'+ tag_dic[key]
    fe = demoji.findall_list(t, desc = True)
    num_emoji = len(fe)
    if encase:
        t = demoji.replace_with_desc(t, sep=tag_dic['emoji'])
    else:
        t = demoji.replace(t, tag_dic['emoji'])

    fet = re.findall(smile_re, t)
    num_smile = len(fet)
    t = re.sub(smile_re, tag_dic['smile'], t)

    filehandles["emoticon_c"]['writer'].write({'ID': id, 'tweetID': tweetId, 'emoji':fe, 'emoticon':fet})


    #have only singular spaces
    t = re.sub(space_collapse_re, " ", t)

    f = re.findall(r"[\d]+", t)
    num_numbers+= len(f)
    t = re.sub(r"[\d]+", tag_dic['number'], t)

    #after length
    num_char_a = len(t)
    filehandles['num']['writer'].write({'ID': id, 'tweetID': tweetId,
                                             'init_len': num_char_b, 'prepr_len': num_char_a,
                   'mentions': num_mention, 'tags': num_tags,'urls':num_urls,
                  'times': num_times, 'emotic_num': num_emoji,
                   'emojis_num': num_smile, 'numericals': num_numbers})



    t = re.sub(not_ascii_re, "", t)
    t = re.sub(space_collapse_re, ' ', t)
    return t

def preprocess(IDs, tweetIDs, tweets, range_CHAR = (2,5),
               range_WORD=(1,5),range_TAG = (1,3),
               range_DEP=(1,5), range_POS = (1,5), spacy=False, both = False, asis=False, encase=['emoji', 'smile'], filehandles=None, num_to_do=0,
               times_done = None, byte=None):

    #this is the preprocessing function for a list of tweets - it then goes on to process every tweet individually
    #expects a list of tweets
    
    #pass uples have the form (ID, processed, Type, range<optional>)
    assert type(filehandles) != type(None)

    tweetsp = []
    for i in range(len(tweets)):
        tweetsp.append(parse_raw(id=IDs[i], tweetId= tweetIDs[i], tweet=tweets[i], spacy=spacy, encase=encase, filehandles=filehandles))


    if spacy:

        tweetsp = parse_spacy(IDs, tweetIDs, tweetsp)
        ranges = {'lemma': range_WORD, 'pos': range_POS, 'tag': range_TAG, 'dep': range_DEP}

        #make the grams
        for i, tweet in enumerate(tweetsp):

            ID = tweet['id']
            tID = tweet['tid']
            for key in ranges.keys():
                grams = make_ngrams_text(tweet[key], ranges[key])

                #make it with immutable so we do not override
                for ln in grams.keys():
                    f = key + '_grams_' + str(ln)
                    filehandles[f]['writer'].write({'ID':ID, 'tweetID': tID, f: grams[ln]})
                    times_done[i] +=1

    if both or not spacy:
        #do other processing as well
        spacy = False
        for i, tweet in enumerate(tweets):
            tweets[i] = parse_raw(id=IDs[i], tweetId=tweetIDs[i], tweet=tweets[i], spacy=spacy, encase=encase,
                                  filehandles=filehandles)
            times_done[i] += 3
            #distortion ngrams
            dist = parse_dist(tweet)
            distgrams = make_ngrams_char(dist, range_CHAR)
            for key in distgrams.keys():
                f = 'dist' + '_grams_' + str(key)
                filehandles[f]['writer'].write({'ID': IDs[i], 'tweetID': tweetIDs[i], f: distgrams[key]})
                times_done[i] += 1


            #chargrams inlduding all special Characters
            if asis:
                words, chars = parse_text_char(tweet, encase, asis)
                chragrams = make_ngrams_char(chars, range_CHAR)
                for key in chragrams.keys():
                    f = 'asis' + '_grams_' + str(key)
                    filehandles[f]['writer'].write({'ID': IDs[i], 'tweetID': tweetIDs[i], f: chragrams[key]})
                    times_done[i] += 1


            # chargrams and wordgrams excluding all special Characters
            words, chars = parse_text_char(tweet, encase)
            chragrams = make_ngrams_char(chars, range_CHAR)
            wordgrams = make_ngrams_text(words, range_WORD)
            for key in chragrams.keys():
                f = 'char' + '_grams_' + str(key)
                filehandles[f]['writer'].write({'ID': IDs[i], 'tweetID': tweetIDs[i], f: chragrams[key]})
                times_done[i] += 1


            for key in wordgrams.keys():
                f = 'word' + '_grams_' + str(key)
                filehandles[f]['writer'].write({'ID': IDs[i], 'tweetID': tweetIDs[i], f: wordgrams[key]})
                times_done[i] += 1

            if times_done[i] >= num_to_do:
                filehandles['process']['writer'].write({'tweetID': tweetIDs[i], 'ID': IDs[i],
                                                        'byte': byte, 'processed': times_done[i]})



def process_wrapper(linebytes, path, filen, savepath, part_id, num_to_do, range_CHAR = (2,5),
               range_WORD=(1,5),range_TAG = (1,3),
               range_DEP=(1,5), range_POS = (1,5), spacy=False,
                    both = False, asis=False, encase=['emoji', 'smile'], done_dic=None, rerun=False):
    #is a wrapper for the ChildDaemon - each daemon writes to its own file - no listener as during tests the queues quickly exploded and the ParentProcess crashed

    pid = mp.current_process()
    print('{} has started to work'.format(pid))
    sys.stdout.flush()
    listener_dic = {'CHAR': range_CHAR,
                    'AsIS': range_CHAR,
                    'TAG': range_TAG,
                    'DEP':range_DEP,
                    'WORD':range_WORD,
                    'LEMMA': range_WORD,
                    'POS': range_POS,
                    'DIST': range_CHAR
    }
    filehandles = file_ceation_wrapper(listener_dic, savepath, part_id, rerun=rerun)
    print(list(filehandles.keys()))
    sys.stdout.flush()


    retweet_re = re.compile(r'(^RT\s{0,1}@[a-zA-Z0-9_]+:{0,1})')
    not_ascii_re = re.compile("([^\x00-\x7F]+)")
    with open(os.path.join(path, filen), 'r', encoding='utf-8') as f:
        for byte in linebytes:
            f.seek(byte)
            line = f.readline()
            if not line:
                continue
            line = ndjson.loads(line)[0]

            ids = []
            tweets = []
            tweetIDs = []
            tweet_counter = 0
            times_done = []
            for tweet in line['text']:
                #filter out the retweets and tweets mostly made of ascii characters (sans emojis)
                if not re.search(retweet_re, tweet) and len(re.sub(not_ascii_re, '', demoji.replace(tweet, repl='P'))) > 10:
                    if done_dic:
                        try:
                            tmp = done_dic.pop('{}_{}'.format(line['id'], tweet_counter))
                        except:
                            ids.append(line['id'])
                            tweets.append(tweet)
                            #make unique tweet ids
                            tweetIDs.append('{}_{}'.format(line['id'], tweet_counter))
                            times_done.append(0)
                    else:
                        ids.append(line['id'])
                        tweets.append(tweet)
                        tweetIDs.append('{}_{}'.format(line['id'], tweet_counter))
                        times_done.append(0)
                    tweet_counter += 1



            if ids:
                preprocess(IDs=ids, tweetIDs=tweetIDs, tweets=tweets, range_CHAR=range_CHAR,
                           range_WORD=range_WORD, range_TAG=range_TAG,
                           range_DEP = range_DEP, range_POS=range_POS, spacy=spacy, both=both, asis=asis ,encase=encase, filehandles=filehandles,
                           num_to_do=num_to_do, times_done=times_done, byte=byte)


            else:
                print('everything already processed...skip')


    print('{} is finished with linebytes {}:{}'.format(pid, linebytes[0], linebytes[-1]))
    sys.stdout.flush()

    for key in filehandles.keys():
        filehandles[key]['writer'].close()
        filehandles[key]['f'].close()

    return 1

def make_files(savepath, typ, ident, rerun=False):
    # create datapaths
    #print('making the file for {} {}'.format(typ, ident))
    #sys.stdout.flush()
    if not os.path.exists(os.path.join(savepath, typ)):
        os.makedirs(os.path.join(savepath, typ))

    # open/create files in append-mode and a create a ndjson-writer
    if rerun:
        mode = 'w'
    else:
        mode = 'a'
    try:
        f = open(os.path.join(savepath, typ, ident), mode=mode, encoding='utf-8')
    except IOError as error:
        print('error opening file with {}'.format(error))
        sys.stdout.flush()

    try:
        writer = jsonlines.Writer(f, flush=True)
    except IOError as error:
        print('error opening file with {}'.format(error))
        sys.stdout.flush()
    #print('made the file for {} {}'.format(typ, ident))
    #print(type(f), type(writer))
    #sys.stdout.flush()
    return f, writer


def file_ceation_wrapper(ranges, savepath, pid, rerun=False):
    typs =['PROCESS', 'POLARITY', "EMOTICON_C", "NUM", 'AsIS',
           'CHAR', 'WORD', 'DIST', 'LEMMA', 'POS', 'TAG', 'DEP']  # 'VECTORS',

    filehandles = {}
    for el in typs:
        #print('el is {}'.format(el))
        sys.stdout.flush()
        if el in ["VECTORS", "NUM", "POLARITY", "EMOTICON_C"]:
            #print('Creating single start', file=o)
            #sys.stdout.flush()
            typ = el.lower()
            featurek = typ
            ident = typ + '_part_{}.ndjson'.format(pid)
            #print('Creating single file handles', file=o)
            sys.stdout.flush()
            f, writer = make_files(savepath, typ, ident, rerun)
            #print('Returned handles', file=o)
            filehandles[featurek] = {}
            #print('Created handles dic', file=o)
            #sys.stdout.flush()
            filehandles[featurek]['f'] = f
            #print('Added filehandle', file=o)
            #sys.stdout.flush()
            filehandles[featurek]['writer'] = writer
            #print('Created and appended single file handles', file=o)
            #sys.stdout.flush()

        elif el == 'PROCESS':
            #print('Creating process', file=o)
            #sys.stdout.flush()
            typ = el.lower()
            featurek = typ
            ident = '_processed_ids_part_{}.ndjson'.format(pid)
            #print('Creating process file handles', file=o)
            #sys.stdout.flush()
            f, writer = make_files(savepath, typ, ident, rerun)
            filehandles[featurek] = {}
            filehandles[featurek]['f'] = f
            filehandles[featurek]['writer'] = writer
            #print('Created and appended process file handles', file=o)
            #sys.stdout.flush()
        else:
            typ = el.lower()
            for i in range(ranges[el][0], ranges[el][1]+1):
                #print('Creating ranges', file=o)
                #sys.stdout.flush()
                featurek = typ + '_grams_' + str(i)
                #print('Created featurek {}'.format(featurek), file=o)
                #sys.stdout.flush()
                ident = typ + '_grams_' + str(i) + '_part_{}.ndjson'.format(pid)
                #print('Created ident {}'.format(ident), file=o)
                #sys.stdout.flush()

                #print('trying to create file for {} {}'.format(typ, i), file=o)
                #sys.stdout.flush()
                f, writer = make_files(savepath, typ, ident, rerun)
                #print('returned filehandles correctly', file=o)
                #sys.stdout.flush()
                filehandles[featurek] = {}
                #print('made dic in file for key {}'.format(featurek), file=o)
                #sys.stdout.flush()
                filehandles[featurek]['f'] = f
                #print('appended f', file=o)
                #sys.stdout.flush()
                filehandles[featurek]['writer'] = writer
                #print('appended writer', file=o)
                #sys.stdout.flush()
                #print('created file for {} {}'.format(typ, i), file=o)
                #sys.stdout.flush()

    return filehandles


def cal_num_comb(listener_dic, spacy, asis, both):

    c = 0
    if not spacy or both:
        c+=3 #for Pol, EMOT, NUM
        for _ in list(range(listener_dic['CHAR'][0], listener_dic['CHAR'][1]))+[1] + \
                 list(range(listener_dic['WORD'][0], listener_dic['WORD'][1]))+[1]+ \
                 list(range(listener_dic['DIST'][0], listener_dic['DIST'][1]))+[1]:
            c+=1

    if spacy:
        for _ in list(range(listener_dic['LEMMA'][0], listener_dic['LEMMA'][1]))+[1] +\
                 list(range(listener_dic['POS'][0], listener_dic['POS'][1]))+[1]+ \
                 list(range(listener_dic['TAG'][0], listener_dic['TAG'][1]))+[1]+ \
                 list(range(listener_dic['DEP'][0], listener_dic['DEP'][1]))+[1]:
            c+=1
        #c+=1 #for vectors

    if asis:
        for _ in list(range(listener_dic['AsIS'][0], listener_dic['AsIS'][1])) + [1]:
            c+=1

    return c

def _main(args):
    #get emoji codes
    if type(None) == type(demoji.last_downloaded_timestamp()):
        demoji.download_codes()

    #these are the variables for the function
    range_CHAR = args['char']
    range_WORD = args['word']
    range_TAG = args['tag']
    range_DEP = args['dep']
    range_POS = args['pos']

    #dic is needed for job amount calculation
    listener_dic = {'CHAR': range_CHAR,
                    'AsIS': range_CHAR,
                    'TAG': range_TAG,
                    'DEP':range_DEP,
                    'WORD':range_WORD,
                    'LEMMA': range_WORD,
                    'POS': range_POS,
                    'DIST': range_CHAR
    }

    c = cal_num_comb(listener_dic, args['spacy'], args['asis'], args['both'])

    savepath = os.path.join(args['save'], args['workset'], args['part'])
    datapath = os.path.join(args['datapath'], args['workset'], args['part'])
    os.makedirs(savepath, exist_ok=True)

    #must use Manager queue here, or will not work
    manager = mp.Manager()
    if not args['test']:
        ncpus = mp.cpu_count()-1
        if ncpus < 80-1:
            print('failed to get 80 cpus - only got {}'.format(ncpus))
        #ncpus = 40-1
    else:
        ncpus = 6
    print(' got {} cpus for calculations'.format(ncpus))





    #load linebytes
    if not os.path.exists(os.path.join(datapath, 'linebytes.json')) or args['rerun']:
        lineBytes = get_linebytes(datapath, args['file'], test=args['test'])
        with open(os.path.join(datapath, 'linebytes.json'), 'w', encoding='utf-8') as f:
            json.dump(lineBytes, f)

    else:
        with open(os.path.join(datapath,'linebytes.json'), 'r', encoding= 'utf-8') as f:
            lineBytes = json.load(f)



    #chunkify list
    lineBytes = chunkify(lineBytes, chunksize=40-5)
    #lineBytes = [[el,] for el in lineBytes]
    #lineBytes =[lineBytes[1]]

    pid = os.getpid()
    py = psutil.Process(pid)
    print('create managed queue...')
    sys.stdout.flush()

    pool = mp.Pool(ncpus, maxtasksperchild=40)
    #pool = mp.Pool(mp.cpu_count()-1, maxtasksperchild=1)
    print('create listener for saving of data...')
    sys.stdout.flush()
    #put listeners to work first

    #fire off workers

    jobs = []
    #create jobs
    print('create jobs')
    sys.stdout.flush()
    runs = 0


    for i, chunk in enumerate(lineBytes):
        # check which we already did in a previous run
        if not args['rerun']:
            done_dic = get_done(savepath, i)
        else:
            done_dic = None
        job = pool.apply_async(process_wrapper,(chunk, datapath, args['file'],
                                                savepath, i, c,range_CHAR,
                                                range_WORD,range_TAG, range_DEP, range_POS,
                                                args['spacy'], args['both'], args['asis'],args['encase_list'],done_dic, args['rerun']))

        jobs.append(job)
        print('''Process memory used by parent after
        async: {}\nVirtual Memory used by parent after async: {}\n\n'''.format(py.memory_info()[0]*9.31*10e-10,py.memory_info()[1]*9.31*10e-10))
        gc.collect()
        runs +=1
    print("made {} tasks to pool".format(runs))

    # collect results from the workers through the pool result queue
    print('collect results from job cycle...')
    sys.stdout.flush()
    for i in range(0, len(jobs)):
        tmp = jobs.pop(0)
        tmp.get()
        del tmp
    print('sleep after cycle...')
    sys.stdout.flush()




    print('closing down the pool and exit :)')
    sys.stdout.flush()
    pool.close()
    pool.join()
    print('done')
    sys.stdout.flush()



if __name__ == "__main__":
    command = True
    if not command:
        args = {}
        args['datapath'] = "../../Data/pan19-celebrity-profiling-training-dataset-2019-01-31"
        args['save'] = "../../Data/pan19-celebrity-profiling-training-dataset-2019-01-31/preprocessed"
        args['file'] ='workset_manager.ndjson'
        args['char'] = (2,5)
        args['word'] =(1,2)
        args['tag'] =(1,3)
        args['dep'] =(1,3)
        args['pos'] =(1,3)
        args ['workset'] ='workset'
        args['part'] ='manager'
        args['rerun'] =True
        args['spacy'] =True
        args['both'] =True
        args['asis'] =True
        args['test'] =True
        args['encase_list'] = ["emoji","emoticon"]



    else:
        #parse arguements
        argparser = argparse.ArgumentParser(description='Arguements for preprocessing and making the ngrams')
        argparser.add_argument_group('required arguments')
        argparser.add_argument('-p', '--datapath', help='Path to parent input directory (relative or absolute)', required=True)
        argparser.add_argument('-f', '--file', help='Name of input file', required=True)
        argparser.add_argument('-s', '--save', help='Path to output directory (relative or absolute)', required=True)
        argparser.add_argument('-c', '--char', help='Range (l,u) for char  ngrams', required=True)
        argparser.add_argument('-w', '--word', help='Range (l,u) for word  ngrams', required=True)
        argparser.add_argument('-t', '--tag', help='Range (l,u) for spacy tag ngrams', required=True)
        argparser.add_argument('-d', '--dep', help='Range (l,u) for spacy dep ngrams', required=True)
        argparser.add_argument('-o', '--pos', help='Range (l,u) for spacy pos ngrams', required=True)

        argparser.add_argument_group('optional arguments')
        argparser.add_argument('--workset', help='Sub-Directory of parent input-directory (if it exists). Helpful if script is executed in loop on many worksets')
        argparser.add_argument('--part', help='Sub-Sub-Directory of parent input-directory (if it exists). Helpful if script is executed in loop on many types')
        argparser.add_argument('--test', help='Set this if it is a testrun only', action='store_true')
        argparser.add_argument('--rerun', help='Set this if it is you want to rerun and ignore old files', action='store_true')
        argparser.add_argument('--spacy', help='Set this if you want to run spacy', action='store_true')
        argparser.add_argument('--both', help='Set this if spacy is set and you want to run the normal ngrams as well', action='store_true')
        argparser.add_argument('--asis', help='Set this if you want to make as-is CHAR ngrams', action='store_true')
        argparser.add_argument('--encase_list',
                               nargs='*',
                               type=str,
                               help='''Set this if you want to encase tags ("url", "hashtag", "mention",
                               "emoji", "emoticon", "time", "number", "date")''')

        #parse arguements
        args = vars(argparser.parse_args())
        # convert to tuples
        for key in ['char', 'word', 'dep', 'tag', 'pos']:
            args[key] = ast.literal_eval(args[key])

    #make windows-unix problem go away
    if  '\\' in args['save']:
        args['datapath'] = os.path.join(*args['datapath'].split('\\'))
    elif '/' in args['save']:
        args['save'] = os.path.join(*args['save'].split('/'))

    if  '\\' in args['datapath']:
        args['datapath'] = os.path.join(*args['datapath'].split('\\'))
    elif '/' in args['datapath']:
        args['datapath'] = os.path.join(*args['datapath'].split('/'))



    args['workset'] = args.get('workset', '')
    args['part'] = args.get('part', '')
    args['test'] = args.get('test', False)
    args['rerun'] = args.get('rerun', False)
    args['spacy'] = args.get('spacy', False)
    args['both'] = args.get('both', False)
    args['asis'] = args.get('asis', False)
    print(args['rerun'])

    tag_dic = {'url': "url",
               'hashtag': "tag",
               'mention': "user",
               'emoji': "emoji",
               'emoticon': "smile",
               'time': "time",
               'number': "number",
               }
    args['encase_list'] =args.get('encase_list', [])
    args['encase_list'] = [tag_dic[el] for el in args['encase_list']]


    _main(args)

