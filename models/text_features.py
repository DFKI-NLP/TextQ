"""
Author: Dinh Nam Pham
dinh_nam.pham[at]dfki[dot]de

This file contains functions to extract (statistical) text features from the text data.
Evry function takes exactly 1 argument (a string) and outputs exactly 1 float value.

All functions are specifically for German, e.g. Flesch Reading Ease is calculated differently than in English.

A list of all function can be obtained by calling the function get_all_text_features().

Functions:
"""

import re
import inspect



#------------------------------------------------------------------------
#Helper function to count syllables in a word
def count_syllables(word: str) -> int:
    word = word.lower()
    syllable_count = 0
    vowels = "aeiouäöü"
    if word[0] in vowels:
        syllable_count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            syllable_count += 1
    if word.endswith("e"):
        syllable_count -= 1
    return max(1, syllable_count)

#------------------------------------------------------------------------

#Average Sentence Length (ASL)
def average_sentence_length(text):
    words = re.findall(r'\b\w+\b', text)
    sentences = re.split(r'[.!?]', text)
    asl = len(words) / len(sentences)
    return float(asl)

#Percentage of Words with at Least Six Letters (PW6)
def percentage_words_six_letters(text):
    words = re.findall(r'\b\w+\b', text)
    pw6 = (sum(1 for word in words if len(word) >= 6) / len(words)) * 100
    return float(pw6)

#Average Number of Syllables per word (ASC)
def average_number_syllables(text):
    words = re.findall(r'\b\w+\b', text)
    syllable_count = sum(count_syllables(word) for word in words)
    asc = syllable_count / len(words)
    return float(asc)

#Percentage of Words with Only One Syllable (PS1)
def percentage_words_one_syllable(text):
    words = re.findall(r'\b\w+\b', text)
    ps1 = (sum(1 for word in words if count_syllables(word) == 1) / len(words)) * 100
    return float(ps1)

#Percentage of Words with at Least Three Syllables (PS3)
def percentage_words_three_syllables(text):
    words = re.findall(r'\b\w+\b', text)
    ps3 = (sum(1 for word in words if count_syllables(word) >= 3) / len(words)) * 100
    return float(ps3)

#Flesch Reading Ease (FRE)
def flesch_reading_ease(text):
    words = re.findall(r'\b\w+\b', text)
    sentences = re.split(r'[.!?]', text)
    asl = len(words) / len(sentences)
    syllable_count = sum(count_syllables(word) for word in words)
    asc = syllable_count / len(words)
    fre = 180 * asl - (58.5 * asc)
    return float(fre)

#Wiener Sachtextformel (WSF-1)
def wiener_sachtextformel_1(text):
    words = re.findall(r'\b\w+\b', text)
    sentences = re.split(r'[.!?]', text)
    asl = len(words) / len(sentences)
    pw6 = (sum(1 for word in words if len(word) >= 6) / len(words)) * 100
    syllable_count = sum(count_syllables(word) for word in words)
    asc = syllable_count / len(words)
    ps1 = (sum(1 for word in words if count_syllables(word) == 1) / len(words)) * 100
    ps3 = (sum(1 for word in words if count_syllables(word) >= 3) / len(words)) * 100
    wstf1 = 0.1935 * ps3 + 0.1672 * asl + 0.1297 * pw6 - 0.0327 * ps1 - 0.875
    return float(wstf1)

#Wiener Sachtextformel (WSF-2)
def wiener_sachtextformel_2(text):
    words = re.findall(r'\b\w+\b', text)
    sentences = re.split(r'[.!?]', text)
    asl = len(words) / len(sentences)
    pw6 = (sum(1 for word in words if len(word) >= 6) / len(words)) * 100
    syllable_count = sum(count_syllables(word) for word in words)
    asc = syllable_count / len(words)
    ps1 = (sum(1 for word in words if count_syllables(word) == 1) / len(words)) * 100
    ps3 = (sum(1 for word in words if count_syllables(word) >= 3) / len(words)) * 100
    wstf2 = 0.2007 * ps3 + 0.1682 * asl + 0.1373 * pw6 - 0.0326 * ps1 - 0.621
    return float(wstf2)

#Wiener Sachtextformel (WSF-3)
def wiener_sachtextformel_3(text):
    words = re.findall(r'\b\w+\b', text)
    sentences = re.split(r'[.!?]', text)
    asl = len(words) / len(sentences)
    pw6 = (sum(1 for word in words if len(word) >= 6) / len(words)) * 100
    syllable_count = sum(count_syllables(word) for word in words)
    asc = syllable_count / len(words)
    ps1 = (sum(1 for word in words if count_syllables(word) == 1) / len(words)) * 100
    ps3 = (sum(1 for word in words if count_syllables(word) >= 3) / len(words)) * 100
    wstf3 = 0.2966 * ps3 + 0.1905 * asl + 0.1539 * pw6 - 0.0455 * ps1 - 1.045
    return float(wstf3)

#Wiener Sachtextformel (WSF-4)
def wiener_sachtextformel_4(text):
    words = re.findall(r'\b\w+\b', text)
    sentences = re.split(r'[.!?]', text)
    asl = len(words) / len(sentences)
    pw6 = (sum(1 for word in words if len(word) >= 6) / len(words)) * 100
    syllable_count = sum(count_syllables(word) for word in words)
    asc = syllable_count / len(words)
    ps1 = (sum(1 for word in words if count_syllables(word) == 1) / len(words)) * 100
    ps3 = (sum(1 for word in words if count_syllables(word) >= 3) / len(words)) * 100
    wstf4 = 0.2744 * ps3 + 0.2656 * asl + 0.2300 * pw6 - 0.0788 * ps1 - 1.370
    return float(wstf4)

#SMOG Index
def smog_index(text):
    words = re.findall(r'\b\w+\b', text)
    sentences = re.split(r'[.!?]', text)
    ps3 = (sum(1 for word in words if count_syllables(word) >= 3) / len(words)) * 100
    smog = 1.0430 * ((ps3 * (30 / len(sentences))) ** 0.5) + 3.1291
    return float(smog)

#Coleman-Liau Index
def coleman_liau_index(text):
    words = re.findall(r'\b\w+\b', text)
    sentences = re.split(r'[.!?]', text)
    l = sum(len(word) for word in words) / len(words) * 100
    s = len(sentences) / len(words) * 100
    cli = 0.0588 * l - 0.296 * s - 15.8
    return float(cli)

#Gunning Fog Index
def gunning_fog_index(text):
    words = re.findall(r'\b\w+\b', text)
    sentences = re.split(r'[.!?]', text)
    ps3 = (sum(1 for word in words if count_syllables(word) >= 3) / len(words)) * 100
    gfi = 0.4 * ((len(words) / len(sentences)) + 100 * (ps3 / len(words)))
    return float(gfi)

#Average chars per word (acpw)
def average_chars_per_word(text):
    words = re.findall(r'\b\w+\b', text)
    chars = sum(len(word) for word in words)
    acpw = chars / len(words)
    return float(acpw)

#Type token ratio: ratio fo different words to total words
def type_token_ratio(text):
    words = re.findall(r'\b\w+\b', text)
    ttr = len(set(words)) / len(words)
    return float(ttr)

#Lexical Density
def lexical_density(text):
    words = re.findall(r'\b\w+\b', text)
    ld = len(set(words)) / len(words)
    return float(ld)

#Number of unique tokens
def number_unique_tokens(text):
    words = re.findall(r'\b\w+\b', text)
    return len(set(words))



#------------------------------------------------------------------------
# Function to get a list with all functions in this file
def get_all_text_features():
    """
    Returns a dictionary where keys are function names and values are the function objects.
    """
    functions = {name: obj for name, obj in globals().items() if inspect.isfunction(obj)}
    #Remove helper functions
    functions.pop("count_syllables")
    functions.pop("get_all_text_features")
    return functions
