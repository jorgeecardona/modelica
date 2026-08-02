# How these pages are written

The documentation follows **ASD-STE100 Simplified Technical English**. STE is a controlled
language. The aerospace industry wrote it in the 1970s to make maintenance manuals clear to
people who do not read English as a first language. It became an international standard in
2025.

Modelica is used all over the world. Most people who read this do not read English first.
STE costs the writer some effort and saves every reader some effort, so the trade is a good
one.

The standard has 53 rules and a dictionary of about 900 approved words. This page lists the
rules that apply here, and the two places this project departs from them.

## The rules

**Sentences**

* Write no more than 25 words in a sentence that describes something.
* Write no more than 20 words in a sentence that tells the reader to do something.
* Give one instruction in one sentence.
* Put the topic at the start of the sentence.

**Paragraphs**

* Write no more than six sentences in a paragraph.
* Give one topic to one paragraph.
* Put the most important sentence first.

**Verbs**

* Use the active voice. Write "the parser reads the file", not "the file is read".
* Use the simple present tense for what the code does now.
* Use the simple past tense for what happened once. Do not use the present perfect.
* Do not make a noun out of a verb when the verb is clearer.

**Words**

* Give one meaning to one word, and use one word for one meaning.
* Use the same word for the same thing every time. Do not look for a synonym.
* Do not use slang, idiom, metaphor or irony.
* Write the article: "the flat model", not "flat model".
* Do not put more than three words together as one noun.

**Structure**

* Use a list when you have more than two parallel items.
* Use a table when each item has the same set of properties.
* Use a diagram when the subject is a shape or an order.

## The two departures

**Technical terms are not in the dictionary, and are used anyway.** STE lets a project add
the words its subject needs. This one needs *flatten*, *causalize*, *index reduction*,
*incidence*, *tearing*, *stiff*, *residual*, *Jacobian* and about twenty more. Each of
them is defined where it first appears. A shorter word does not exist for any of them.

**A design note may argue.** Most pages tell the reader what the software does. Some pages
say why one design was chosen and another was refused. Those pages still obey every rule
above. They contain more sentences that start with "because" and "so", because a reason is
what they carry.

## What this is not

STE is about the sentences, not about the content. It does not permit a page to say less.
Every claim, every number and every reason that was in these pages before the rewrite is
still in them. The sentences that hold them are shorter.
