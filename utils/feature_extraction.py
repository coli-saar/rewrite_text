"""
List of features supported:
1) maximum dependency length  SYNTACTIC COMPLEXITY OF THE TARGET
2) target/source ratio in terms of character count  LENGTH RATIO
3) Levenshtein edit ratio: between 0 and 1  SURFACE FORM SIMILARITY
4) word frequency in the target: third quantile of the word frequencies in the target sentence


"""

import spacy
import numpy as np
import string
import re
import Levenshtein
from nltk.corpus import stopwords
#from utils.prepare_word_embeddings_frequency_ranks import load_ranks
from utils.feature_bin_preparation import create_bins


def walk_tree(node, depth):
    """ Pass a spacy root of a sentence, return the maximum length """
    if node.n_lefts + node.n_rights > 0:
        return max(walk_tree(child, depth + 1) for child in node.children)
    else:
        return depth


def maximum_dependency_length(s, t, lang, absolute):
    """ Language-dependent, language model loaded from spacy, parse
    Calculate the maximum dependency length of the SOURCE (complex) and divide it by that of TARGET (simple)
    """
    lang_model = {"en": "en_core_web_sm", "de": "de_core_news_sm"}
    if lang.lower() not in lang_model:
        print("Language choice not supported, defaulting to English (other option: German)")
        lang = "en"

    nlp_model = spacy.load(lang_model[lang])
    doc = nlp_model(t)  # target
    #all_depths = [walk_tree(sent.root, 0) for sent in doc.sents]
    #print(all_depths)
    max_depth_target = max([walk_tree(sent.root, 0) for sent in doc.sents])

    if absolute:
        return max_depth_target

    doc = nlp_model(s)  # source
    #all_depths = [walk_tree(sent.root, 0) for sent in doc.sents]
    #print(all_depths)
    max_depth_source = max([walk_tree(sent.root, 0) for sent in doc.sents])

    if max_depth_source == 0:  # single word sentences...
        #print("source depth 0??")
        #import pdb; pdb.set_trace()
        max_depth_source = 0.5
    if max_depth_target == 0:
        #print("target depth 0??")
        #import pdb; pdb.set_trace()
        max_depth_target = 0.5
    depth_ratio = max_depth_target / max_depth_source

    return depth_ratio


def character_length_ratio(_src, _tgt, absolute):
    """ Calculate the ratio between target and source length in characters
    len(simple) / len(complex)  OR len(target) / len(source)
    """
    if absolute:
        return len(_tgt)
    return len(_tgt)/len(_src)


def Levenshtein_ratio(c, s):
    """ Calculate the Levenshtein ratio between the source and target: the ratio is normalized, 0-1
    Note that this ratio is symmetric: c->s and s->c gives the same score
    Higher score = higher similarity
    """
    return Levenshtein.ratio(c, s)


def get_freq_rank_word(word, ranks_dict):
    """ If the words in is vocabulary, return its rank (int), else return the very final rank """
    if word in ranks_dict:
        return ranks_dict[word] + 1
    return len(ranks_dict) + 1


def get_log_freq_rank_word(word, ranks):
    """ If the words in is vocabulary, return its log rank (int) with natural log as base,
    else return the log of the very final rank """
    if word in ranks:
        return np.log(ranks[word] + 1)
    if word == "-NO-WORDS-":
        return np.log(len(ranks) + 1)
    return np.log(len(ranks) + 1)


def word_checks(token):
    """
    :param token: a string of the token in sentence
    :return: Boolean
    This function check if the token is a punctuation symbol or a number. If any of the two, return False
    """
    is_int_or_float = re.compile(r"^[+-]?((\d+(\.\d+)?)|(\.\d+))$")  # should match with "1", "3.68", but NOT "3s1"
    if token in string.punctuation:
        return False
    if is_int_or_float.match(token):
        return False
    return True


def stop_word_check(token, lang):
    if lang == "de":
        _stopwords = stopwords.words("german")
    else:
        _stopwords = stopwords.words("english")
    if token in _stopwords:
        return False
    return True


def properties_word_freq_in_sentence(frequency_ranks, all_ranks):
    """
    :param frequency_ranks: a list of float/int frequency ranks from selected words in sentence
    :param all_ranks: a dictionary of ranks of word frequencies
    :return: third quantile (the value right between the median and the max)
    """
    #mu = round(np.mean(frequency_ranks), 2)
    #mode = round(np.max(frequency_ranks), 2)
    #sd = np.std(frequency_ranks)
    #medi = np.median(frequency_ranks)
    if not frequency_ranks:  # if the list of ranks is emtpy because all words were numbers or stopwords or punct.
        frequency_ranks = [get_log_freq_rank_word("-NO-WORDS-", all_ranks)]

    #first_quantile = np.quantile(frequency_ranks, 0.25)
    third_quantile = np.quantile(frequency_ranks, 0.75)
    #q3q1 = third_quantile - first_quantile

    # print("Mean %f, Mode %f, Median %f, SD %f, Q1 %f, Q3 %f, Q3-Q1 %f" % (mu, mode, medi, sd, first_quantile,
    #                                                                       third_quantile, q3q1))
    return third_quantile


def word_frequency_rank(source, target, lang, _ranks, absolute):
    """
    Each word is associated with a frequency, For a sentence we get a distribution of frequencies.
    Properties of a distribution: mean, standard dev, quartiles
    Think about using log of the frequency rank
    """
    # step 0: load the rank dictionary
    #ranks = load_ranks(lang)

    # step 1: tokenize
    lang_model = {"en": "en_core_web_sm", "de": "de_core_news_sm"}
    if lang.lower() not in lang_model:
        print("Language choice not supported, defaulting to English (other option: German (de))")
        lang = "en"

    nlp_model = spacy.load(lang_model[lang])
    doc_target = nlp_model(target)

    # step 2: get the (log) frequency rank for each token, observe for entire sentence
    # but ignore punctuation and numbers as well as stopwords
    considered_tokens = [w.text for w in doc_target if word_checks(w.text) and stop_word_check(w.text, lang)]
    # target_ranks = [get_freq_rank_word(w.text, ranks) for w in doc if word_checks(w.text) and
    #                 stop_word_check(w.text, lang)]
    target_ranks = [get_log_freq_rank_word(w.text, _ranks) for w in doc_target if word_checks(w.text) and
                    stop_word_check(w.text, lang)]
    third_quantile_target = properties_word_freq_in_sentence(target_ranks, _ranks)

    if absolute:
        return third_quantile_target
    # repeat the process for the source and return the ratio
    doc_source = nlp_model(source)
    source_ranks = [get_log_freq_rank_word(w.text, _ranks) for w in doc_source if word_checks(w.text) and
                    stop_word_check(w.text, lang)]
    third_quantile_source = properties_word_freq_in_sentence(source_ranks, _ranks)

    return third_quantile_target / third_quantile_source


def feature_bundle_sentence(source, target, lang, list_of_desired_features, _freq_ranks):
    bundle = {}
    if "frequency" in list_of_desired_features:
        v = word_frequency_rank(source, target, lang, _freq_ranks, absolute=False)  # TODO: fix ranks: should be an arg
        bundle["frequency"] = v
    if "dependency" in list_of_desired_features:
        v = maximum_dependency_length(source, target, lang, absolute=False)
        bundle["dependency"] = v
    if "length" in list_of_desired_features:
        v = character_length_ratio(source, target, absolute=False)
        bundle["length"] = v
    if "levenshtein" in list_of_desired_features:
        v = Levenshtein_ratio(source, target)
        bundle["levenshtein"] = v
    return bundle


def get_bin_value(x_val, this_feature_bins):
    # using numpy digitize
    idx = np.digitize([x_val], this_feature_bins, right=True).item()
    if idx == len(this_feature_bins):  # the idx is out of range if the value is bigger than the last bin
        idx -= 1
    return this_feature_bins[idx]


def feature_bins_bundle_sentence(source, target, lang, list_of_desired_features, feature_bin_dictionary, _freq_ranks):
    bundle = {}
    bundle_exact = {}
    if "frequency" in list_of_desired_features:
        v = word_frequency_rank(source, target, lang, _freq_ranks, absolute=False)
        v_bin = get_bin_value(v, feature_bin_dictionary["frequency"])
        bundle["frequency"] = v_bin
        bundle_exact["frequency"] = v
    if "dependency" in list_of_desired_features:
        v = maximum_dependency_length(source, target, lang, absolute=False)
        v_bin = get_bin_value(v, feature_bin_dictionary["dependency"])
        bundle["dependency"] = v_bin
        bundle_exact["dependency"] = v
    if "length" in list_of_desired_features:
        v = character_length_ratio(source, target, absolute=False)
        v_bin = get_bin_value(v, feature_bin_dictionary["length"])
        bundle["length"] = v_bin
        bundle_exact["length"] = v
    if "levenshtein" in list_of_desired_features:
        v = Levenshtein_ratio(source, target)
        v_bin = get_bin_value(v, feature_bin_dictionary["levenshtein"])
        bundle["levenshtein"] = v_bin
        bundle_exact["levenshtein"] = v
    return bundle, bundle_exact


#### TESTING ###

features = ["frequency", "dependency", "length", "levenshtein"]

sources_de = ["Gouverneur Gavin Newsom rief wegen der Feuer und der anhaltenden Sommerhitze am Dienstag den Notstand aus .",
"Eine Reihe von Dörfern in Kärnten war am Dienstagnachmittag noch immer von der Außenwelt abgeschnitten .",
"Damit gab es zwar ein Wachstum , dieses hat sich aber deutlich verlangsamt .",
"Auf Twitter entschuldigte sich Van der Bellen am Sonntagabend ebenfalls noch einmal für seinen nächtlichen Besuch im Gastgarten eines Lokals weit nach der Corona-Sperrstunde .",
"Sonst bleibt es noch länger sonnig und verbreitet trocken .",
"Ein anderes Feuer , das in einem sogenannten Satellitenlager außerhalb des Registrierlagers von Moria ausgebrochen war , konnte die Feuerwehr unter Einsatz eines Löschflugzeugs schnell löschen , wie das Fernsehen zeigte .",
"In vier der fünf erfassten österreichischen Städte hat die Zeit im Stau gegenüber 2018 übrigens zugenommen , lediglich Graz blieb auf dem Niveau von 2018 .",
"Die geplanten Teilnahmen in den aktuell laufenden Kurzarbeitsprojekten beliefen sich laut Ministerium Ende September auf 295.200 Mitarbeiter , um rund 157.300 Personen weniger als noch Ende August .",
"Nach einer Zwischenkundgebung bei der Wirtschaftskammer Wien soll um 14.00 Uhr die Abschlusskundgebung vor den Ministerien für Nachhaltigkeit und Wirtschaft am Stubenring abgehalten werden .",
"Ganz allgemein können wir beobachten , dass der Druck auf Kinder und Jugendliche in den letzten Jahren immer weiter gestiegen ist   , berichtete Satke .",
"Der Mann konnte ohne Bewusstsein von anderen Badegästen zwar kurz darauf aus dem See gezogen werden , die Reanimationsversuche durch den alarmierten Notarzt blieben jedoch erfolglos .",
"Rund 650.000 Schüler in Vorarlberg , Tirol , Salzburg , Oberösterreich , Steiermark und Kärnten erhalten am Freitag ihre Zeugnisse und starten in die neunwöchigen Sommerferien .",
"In Österreich haben sich unterdessen binnen 24 Stunden erneut mehr als hundert Menschen infiziert .",
"So lebte die Chance für Rapid , Salzburg nach dem 2 : 0 im Ligaspiel Ende Februar ein zweites Mal in der laufenden Saison in die Knie zu zwingen .",
"Leonhard Schitter , Präsident von Oesterreichs Energie , betonte in einer Aussendung die Freiwilligkeit der Lösung :   So können wir unseren Kunden unter die Arme greifen , wenn sie es am meisten brauchen - schnell und unbürokratisch .",
"Die Zuschläge von 100 Prozent auf die 11. und 12. Arbeitsstunde entsprächen der   österreichischen Logik   der Arbeitnehmer :   Wir sind durchaus bereit , etwas mehr zu arbeiten , aber dafür wollen wir auch etwas haben .",
"Der Grund : Im Abspann des Films bedanken sich die Filmemacher bei Sicherheitsbehörden in der westchinesischen Region Xinjiang , wo nach Angaben von Menschenrechtlern seit Jahren Hunderttausende Mitglieder der muslimischen Minderheit der Uiguren in Umerziehungslagern festgehalten werden .",
"Neu dazu kommt nun auch eine Regelung für Fahrgemeinschaften zwischen Personen , die nicht im selben Haushalt leben : Diese sind nur mit Maske und ein Meter Abstand zulässig .",
"Ein starkes Erdbeben hat am Dienstag gegen 12.20 Uhr Zentralkroatien erschüttert und bisher sieben Menschenleben und zahlreiche Verletzte gefordert .",
"Brasiliens Präsident Jair Bolsonaro hatte sich darüber empört , dass die G7 -Staaten sich in die inneren Angelegenheiten Brasiliens einmischten .",
"Damit werden die Boliden erstmals seit 1954 nicht auf dem Stadtkurs ihre Runden drehen .",
"Die Hochrechnung ( Auszählungsgrad 100 Prozent ) beinhaltet auch schon eine Briefwahl-Schätzung .",
"SPÖ-Chefin Pamela Rendi-Wagner geht in die Offensive und lässt die Parteibasis entscheiden , ob sie an der Spitze der Sozialdemokratie bleiben soll .",
"Die Infektionszahlen würden durch falsche Sicherheit wieder in die Höhe gehen .",
"Laut Aussendung der Veranstalter gingen die drei mit je 1.000 Euro dotierten Hauptpreise an Autoren ,   die über das Verletzt sein und den Mut schreiben und uns in eine neue Welt voller Zahlen und Buchstaben eintauchen lassen   ."]

targets_de = ["Während der Sommerhitze und der Trockenheit sind Tausende Feuerwehrleute in mehreren Regionen im Einsatz .",
"Die Bewohner von 15 Häusern mussten in Sicherheit gebracht werden .",
"2018 waren es ein bisschen weniger .",
"Bundespräsident Van der Bellen hielt sich nicht an Lokal-Sperrstunde",
"Am Wochenende wird es in Österreich noch heißer",
"Wie das Feuer ausbrach , weiß man noch nicht .",
"In Wien hat es im Jahr 2019 von allen Städten in Österreich die meisten Staus gegeben .",
"Das waren um 157.300 weniger als noch Ende August .",
"Durch den Klima-Wandel verändern sich die Temperaturen auf der Erde mehr als es die Natur verkraftet .",
"Suizid ist einer der häufigsten Gründe für den Tod von jungen Menschen .",
"Badegäste zogen den Mann aus dem Wasser .",
"In den Bundesländern Vorarlberg , Salzburg , Tirol , Oberösterreich , Steiermark und Kärnten ist jetzt 9 Wochen lang schulfrei .",
"Innerhalb von 24 Stunden infizierten sich mehr als 100 Menschen mit dem Virus .",
"Der ÖFB-Cup ist ein Fußball-Bewerb , bei dem Mannschaften aus verschiedenen österreichischen Fußball-Ligen gegeneinander spielen .",
"Deswegen wird es für viele schwierig die Miete und Dinge wie Strom , Gas und Wärme zu bezahlen .",
"In der Metall-Industrie haben sich am Sonntag die Gewerkschaft und die Arbeit-Geber auf eine Lohn-Erhöhung geeinigt .",
"Diese Menschen gehören zum muslimischen Volk der Uiguren .",
"Draußen muss man aber mindestens einen Meter Abstand zu anderen Menschen halten .",
"Dabei starben mindestens 7 Menschen und viele Menschen wurden verletzt .",
"Brasilien zeigt sich zwar dankbar für die angebotene Hilfe .",
"Damit wird zum ersten Mal seit dem Jahr 1954 nicht in Monaco gefahren .",
"Die EU-Wahl wird auch Europa-Wahl genannt .",
"Sie findet , das Vertrauen der Mitglieder ist wichtig für sie selbst und die Partei .",
"Für die Impfungen in Österreich sollen 16,5 Millionen Impfungen gekauft werden .",
"Der Preis-Richter und Schriftsteller Felix Mitterer sagte :   Eines haben wir inzwischen gelernt über euch , Ihr alle seid mutiger als viele da draußen   ."]

# corpus_features = {f: [] for f in features}
# feature_bins = create_bins()
#
# for src_sent, tgt_sent in zip(sources_de, targets_de):
#     pair_features = feature_bins_bundle_sentence(src_sent, tgt_sent, "de", [], features, feature_bins)
#     for f, v in pair_features.items():
#         corpus_features[f].append(v)
#
# for f, v in corpus_features.items():
#     # print(f, sorted(v))
#     print(f)
#     print(np.min(v), np.max(v), np.mean(v), np.std(v))
#     print("\n"*2)
