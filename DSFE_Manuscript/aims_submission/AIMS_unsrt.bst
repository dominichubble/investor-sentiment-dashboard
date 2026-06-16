% BibTeX standard bibliography style `unsrt'
   % Version 0.99b (8-Dec-10 release) for BibTeX versions 0.99a or later.
   % Copyright (C) 1984, 1985, 1988, 2010 Howard Trickey and Oren Patashnik.
   % Unlimited copying and redistribution of this file are permitted as long as
   % it is unmodified.  Modifications (and redistribution of modified versions)
   % are also permitted, but only if the resulting file is renamed to something
   % besides btxbst.doc, plain.bst, unsrt.bst, alpha.bst, and abbrv.bst.
   % This restriction helps ensure that all standard styles are identical.
   % The file btxbst.doc has the documentation for this style.
FUNCTION {bbl.and}    {"and"}
FUNCTION {bbl.etal}   { "et~al." }
FUNCTION {bbl.editors}{ "eds." }
FUNCTION {bbl.editor} { "ed." }
FUNCTION {bbl.edby}   { "edited by" }

FUNCTION {bbl.edition}{ "edition" }
FUNCTION {bbl.volume} { "vol." }
FUNCTION {bbl.of}     { "of" }
FUNCTION {bbl.number} { "no." }
FUNCTION {bbl.nr}     { "no." }
FUNCTION {bbl.in}     { "in" }
FUNCTION {bbl.pages}  { "" }
FUNCTION {bbl.page}   { "" }
FUNCTION {bbl.chapter}  { "chapter" }
FUNCTION {bbl.techrep}  { "Technical Report" }
%FUNCTION {bbl.techrep}  { "Tech. Rep." }
FUNCTION {bbl.mthesis}  { "Master's thesis" }
FUNCTION {bbl.phdthesis}{ "PhD thesis" }
%FUNCTION {bbl.phdthesis}{ "Ph.D. thesis" }
FUNCTION {bbl.first}  { "1st" }
FUNCTION {bbl.second} { "2nd" }
FUNCTION {bbl.third}  { "3rd" }
FUNCTION {bbl.fourth} { "4th" }
FUNCTION {bbl.fifth}  { "5th" }
FUNCTION {bbl.st}{ "st" }
FUNCTION {bbl.nd}{ "nd" }
FUNCTION {bbl.rd}{ "rd" }
FUNCTION {bbl.th}{ "th" }

ENTRY
  { address
    arxiv
    author
    booktitle
    chapter
    doi
    edition
    editor
    howpublished
    institution
    journal
    key
    month
    note
    number
    organization
    pages
    publisher
    school
    series
    title
    type
    url
    volume
    year
  }
  {}
  { label }

INTEGERS { output.state before.all mid.sentence after.sentence after.block }

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
}

STRINGS { s t }

FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ }
    { output.state after.block =
        { ", " * write$
          newline$
          "\newblock " write$
        }
        { output.state before.all =
            'write$
            { add.period$ " " * write$ }
          if$
        }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}
%FUNCTION {output.nonnull}
%{ 's :=
%  output.state mid.sentence =
%    { ", " * write$ }
%    { output.state after.block =
%        { add.period$ write$
%          newline$
%          "\newblock " write$
%        }
%        { output.state before.all =
%            'write$
%            { add.period$ " " * write$ }
%          if$
%        }
%      if$
%      mid.sentence 'output.state :=
%    }
%  if$
%  s
%}

FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}

FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$ }
    'output.nonnull
  if$
}

FUNCTION {output.bibitem}
{ newline$
  "\bibitem{" write$
  cite$ write$
  "}" write$
  newline$
  "\newblock "
  before.all 'output.state :=
}

FUNCTION {fin.entry}
{ add.period$
  write$
  newline$
}

FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}

FUNCTION {new.sentence}
{ output.state after.block =
    'skip$
    { output.state before.all =
        'skip$
        { after.sentence 'output.state := }
      if$
    }
  if$
}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}

FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}

FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}

FUNCTION {new.block.checka}
{ empty$
    'skip$
    'new.block
  if$
}

FUNCTION {new.block.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.block
  if$
}

FUNCTION {new.sentence.checka}
{ empty$
    'skip$
    'new.sentence
  if$
}

FUNCTION {new.sentence.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.sentence
  if$
}

FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}

FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { "{\em " swap$ * "}" * }
  if$
}

FUNCTION {bolden}
{ duplicate$ empty$
    { pop$ "" }
    { "\textbf{" swap$ * "}" * }
  if$
}

FUNCTION {tie.or.space.prefix}
{ duplicate$ text.length$ #3 <
    { "~" }
    { " " }
  if$
  swap$
}

FUNCTION {capitalize}
{ "u" change.case$ "t" change.case$ }

FUNCTION {space.word}
{ " " swap$ * " " * }

FUNCTION {eng.ord}
{ duplicate$ "1" swap$ *
  #-2 #1 substring$ "1" =
     { bbl.th * }
     { duplicate$ #-1 #1 substring$
       duplicate$ "1" =
         { pop$ bbl.st * }
         { duplicate$ "2" =
             { pop$ bbl.nd * }
             { "3" =
                 { bbl.rd * }
                 { bbl.th * }
               if$
             }
           if$
          }
       if$
     }
   if$
}

FUNCTION {bibinfo.check}
{ swap$
  duplicate$ missing$
    {
      pop$ pop$
      ""
    }
    { duplicate$ empty$
        {
          swap$ pop$
        }
        { swap$
          pop$
        }
      if$
    }
  if$
}

FUNCTION {bibinfo.warn}
{ swap$
  duplicate$ missing$
    {
      swap$ "missing " swap$ * " in " * cite$ * warning$ pop$
      ""
    }
    { duplicate$ empty$
        {
          swap$ "empty " swap$ * " in " * cite$ * warning$
        }
        { swap$
          pop$
        }
      if$
    }
  if$
}

FUNCTION {format.url}
{
  url
  duplicate$ empty$
    { pop$ "" }
    { "\urlprefix\url{" swap$ * "}" * }
  if$
}


FUNCTION {format.arxiv}
{ arxiv empty$
    { "" }
	{"\arxiv{" arxiv * "}" * }
  if$
}

INTEGERS { nameptr namesleft numnames }

STRINGS  { bibinfo}

FUNCTION {format.names}
{ 'bibinfo :=
  duplicate$ empty$ 'skip$ {
  's :=
  "" 't :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr
      "{ff~}{vv~}{ll}{~jj}"
      format.name$
      bibinfo bibinfo.check
      't :=
      nameptr #1 >
        {
          namesleft #1 >
            { ", " * t * }
            {
              s nameptr "{ll}" format.name$ duplicate$ "others" =
                { 't := }
                { pop$ }
              if$
              t "others" =
                {
                  " " * bbl.etal *
                }
                {
                  bbl.and
                  space.word * t *
                }
              if$
            }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
  } if$
}


FUNCTION {format.names.abbrv}
{ 'bibinfo :=
  duplicate$ empty$ 'skip$ {
  's :=
  "" 't :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr
      "{f.~}{vv~}{ll}{~jj}"
      format.name$
      bibinfo bibinfo.check
      't :=
      nameptr #1 >
        {
          namesleft #1 >
            { ", " * t * }
            {
              s nameptr "{ll}" format.name$ duplicate$ "others" =
                { 't := }
                { pop$ }
              if$
              t "others" =
                {
                  " " * bbl.etal *
                }
                {
                  bbl.and
                  space.word * t *
                }
              if$
            }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
  } if$
}

FUNCTION {format.names.ed}
{
  format.names.abbrv
}

FUNCTION {format.authors}
{ author "author" format.names.abbrv }

FUNCTION {get.bbl.editor}
{ editor num.names$ #1 > 'bbl.editors 'bbl.editor if$ }

FUNCTION {format.editors}
{ editor "editor" format.names duplicate$ empty$ 'skip$
    {
      " " *
      get.bbl.editor
   "(" swap$ * ")" *
      *
    }
  if$
}

FUNCTION {format.note}
{
 note empty$
    { "" }
    { note #1 #1 substring$
      duplicate$ "{" =
        'skip$
        { output.state mid.sentence =
          { "l" }
          { "u" }
        if$
        change.case$
        }
      if$
      note #2 global.max$ substring$ * "note" bibinfo.check
    }
  if$
}

FUNCTION {format.title}
{ title
  duplicate$ empty$ 'skip$
    { "t" change.case$ }
  if$
  "title" bibinfo.check
}

%FUNCTION {format.authors}
%{ author empty$
%    { "" }
%    { author format.names }
%  if$
%}

%FUNCTION {format.editors}
%{ editor empty$
%    { "" }
%    { editor format.names
%      editor num.names$ #1 >
%        { ", editors" * }
%        { ", editor" * }
%      if$
%    }
%  if$
%}
%
%FUNCTION {format.title}
%{ title empty$
%    { "" }
%    { title "t" change.case$ }
%  if$
%}

FUNCTION {n.dashify}
{ 't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
        { t #1 #2 substring$ "--" = not
            { "--" *
              t #2 global.max$ substring$ 't :=
            }
            {   { t #1 #1 substring$ "-" = }
                { "-" *
                  t #2 global.max$ substring$ 't :=
                }
              while$
            }
          if$
        }
        { t #1 #1 substring$ *
          t #2 global.max$ substring$ 't :=
        }
      if$
    }
  while$
}

FUNCTION {word.in}
{ bbl.in %capitalize
  " " * }

FUNCTION {format.date}
{
  ""
  duplicate$ empty$
  year  "year"  bibinfo.check duplicate$ empty$
    { swap$ 'skip$
        { "there's a month but no year in " cite$ * warning$ }
      if$
      *
    }
    { swap$ 'skip$
        {
          swap$
          " " * swap$
        }
      if$
      *
    }
  if$
  duplicate$ empty$
    'skip$
    {
      before.all 'output.state :=
    ", " swap$ *
%    " (" swap$ * ")" *
    }
  if$
}
%FUNCTION {format.date}
%{ year empty$
%    { month empty$
%        { "" }
%        { "there's a month but no year in " cite$ * warning$
%          month
%        }
%      if$
%    }
%    { month empty$
%        'year
%        { month " " * year * }
%      if$
%    }
%  if$
%}

FUNCTION{format.year}
{ year "year" bibinfo.check duplicate$ empty$
    {  "empty year in " cite$ *
       warning$
    }
    {
  "(" swap$ * ")" *
    }
  if$
}
%FUNCTION{format.year}
%{ year "year" bibinfo.check duplicate$ empty$
%    {  "empty year in " cite$ *
%       warning$
%    }
%    {
%  "(" swap$ * ")" *
%    }
%  if$
%}

FUNCTION {format.btitle}
{ title "title" bibinfo.check
  duplicate$ empty$ 'skip$
    {
      emphasize
    }
  if$
}
%FUNCTION {format.btitle}
%{ title emphasize
%}

FUNCTION {tie.or.space.connect}
{ duplicate$ text.length$ #3 <
    { "~" }
    { " " }
  if$
  swap$ * *
}

FUNCTION {either.or.check}
{ empty$
    'pop$
    { "can't use both " swap$ * " fields in " * cite$ * warning$ }
  if$
}

FUNCTION {format.bvolume}
{ volume empty$
    { "" }
    { bbl.volume volume tie.or.space.prefix
      "volume" bibinfo.check * *
      series "series" bibinfo.check
      duplicate$ empty$ 'pop$
        { swap$ bbl.of space.word * swap$ * }
      if$
      "volume and number" number either.or.check
    }
  if$
}

FUNCTION {format.bvolume.bold}
{ volume empty$
    { "" }
    { bbl.volume volume tie.or.space.prefix
      bolden "volume" bibinfo.check * *
      series "series" bibinfo.check
      duplicate$ empty$ 'pop$
        { swap$ bbl.of space.word * swap$ * }
      if$
      "volume and number" number either.or.check
    }
  if$
}

FUNCTION {format.number.series}
{ volume empty$
    { number empty$
        { series field.or.null }
        { series empty$
            { number "number" bibinfo.check }
            { bbl.number
              number tie.or.space.prefix "number" bibinfo.check * *
              bbl.in space.word *
              series "series" bibinfo.check *
            }
          if$
        }
      if$
    }
    { "" }
  if$
}

%FUNCTION {format.bvolume}
%{ volume empty$
%    { "" }
%    { "volume" volume tie.or.space.connect
%      series empty$
%        'skip$
%        { " of " * series emphasize * }
%      if$
%      "volume and number" number either.or.check
%    }
%  if$
%}
%
%FUNCTION {format.number.series}
%{ volume empty$
%    { number empty$
%        { series field.or.null }
%        { output.state mid.sentence =
%            { "number" }
%            { "Number" }
%          if$
%          number tie.or.space.connect
%          series empty$
%            { "there's a number but no series in " cite$ * warning$ }
%            { " in " * series * }
%          if$
%        }
%      if$
%    }
%    { "" }
%  if$
%}

FUNCTION {is.num}
{ chr.to.int$
  duplicate$ "0" chr.to.int$ < not
  swap$ "9" chr.to.int$ > not and
}

FUNCTION {extract.num}
{ duplicate$ 't :=
  "" 's :=
  { t empty$ not }
  { t #1 #1 substring$
    t #2 global.max$ substring$ 't :=
    duplicate$ is.num
      { s swap$ * 's := }
      { pop$ "" 't := }
    if$
  }
  while$
  s empty$
    'skip$
    { pop$ s }
  if$
}

FUNCTION {convert.edition}
{ extract.num "l" change.case$ 's :=
  s "first" = s "1" = or
    { bbl.first 't := }
    { s "second" = s "2" = or
        { bbl.second 't := }
        { s "third" = s "3" = or
            { bbl.third 't := }
            { s "fourth" = s "4" = or
                { bbl.fourth 't := }
                { s "fifth" = s "5" = or
                    { bbl.fifth 't := }
                    { s #1 #1 substring$ is.num
                        { s eng.ord 't := }
                        { edition 't := }
                      if$
                    }
                  if$
                }
              if$
            }
          if$
        }
      if$
    }
  if$
  t
}

FUNCTION {format.edition}
{ edition duplicate$ empty$ 'skip$
    {
      convert.edition
      output.state mid.sentence =
        { "l" }
        { "t" }
      if$ change.case$
      "edition" bibinfo.check
      " " * bbl.edition *
    }
  if$
}

%FUNCTION {format.edition}
%{ edition empty$
%    { "" }
%    { output.state mid.sentence =
%        { edition "l" change.case$ " edition" * }
%        { edition "t" change.case$ " edition" * }
%      if$
%    }
%  if$
%}

INTEGERS { multiresult }

FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
    { multiresult not
      t empty$ not
      and
    }
    { t #1 #1 substring$
      duplicate$ "-" =
      swap$ duplicate$ "," =
      swap$ "+" =
      or or
        { #1 'multiresult := }
        { t #2 global.max$ substring$ 't := }
      if$
    }
  while$
  multiresult
}

FUNCTION {format.pages}
{ pages duplicate$ empty$ 'skip$
    { duplicate$ multi.page.check
        {
          n.dashify
        }
        {
        }
      if$
      "pages" bibinfo.check
    }
  if$
}

%FUNCTION {format.pages}
%{ pages empty$
%    { "" }
%    { pages multi.page.check
%        { "pages" pages n.dashify tie.or.space.connect }
%        { "page" pages tie.or.space.connect }
%      if$
%    }
%  if$
%}

FUNCTION {format.journal.pages}
{ pages duplicate$ empty$ 'pop$
    { swap$ duplicate$ empty$
        { pop$ pop$ format.pages }
        {
%          ", " *
          " " *
%          space.word
          format.year * ", " *
          swap$
          n.dashify
          "pages" bibinfo.check
          *
        }
      if$
    }
  if$
}

FUNCTION {format.vol.num.pages}
{ volume field.or.null
  duplicate$ empty$ 'skip$
    {
      "volume" bibinfo.check
    }
  if$
  bolden
  format.journal.pages
}

%FUNCTION {format.vol.num.pages}
%{ volume field.or.null
%  number empty$
%    'skip$
%    { "(" number * ")" * *
%      volume empty$
%        { "there's a number but no volume in " cite$ * warning$ }
%        'skip$
%      if$
%    }
%  if$
%  pages empty$
%    'skip$
%    { duplicate$ empty$
%        { pop$ format.pages }
%        { ":" * pages n.dashify * }
%      if$
%    }
%  if$
%}

FUNCTION {format.chapter.pages}
{ chapter empty$
    'format.pages
    { type empty$
        { bbl.chapter }
        { type "l" change.case$
          "type" bibinfo.check
        }
      if$
      chapter tie.or.space.prefix
      "chapter" bibinfo.check
      * *
      pages empty$
        'skip$
        { ", " * format.pages * }
      if$
    }
  if$
}
%FUNCTION {format.chapter.pages}
%{ chapter empty$
%    'format.pages
%    { type empty$
%        { "chapter" }
%        { type "l" change.case$ }
%      if$
%      chapter tie.or.space.connect
%      pages empty$
%        'skip$
%        { ", " * format.pages * }
%      if$
%    }
%  if$
%}

FUNCTION {format.booktitle}
{
  booktitle "booktitle" bibinfo.check
  emphasize
}
FUNCTION {format.in.ed.booktitle}
{ format.booktitle duplicate$ empty$ 'skip$
    {
      editor "editor" format.names.ed duplicate$ empty$ 'pop$
        {
          get.bbl.editor
          " " * swap$ *
          "(" swap$ * ")" *
          swap$
          " " * swap$
          * }
      if$
      word.in swap$ *
    }
  if$
}
%FUNCTION {format.in.ed.booktitle}
%{ booktitle empty$
%    { "" }
%    { editor empty$
%        { "In " booktitle emphasize * }
%        { "In " format.editors * ", " * booktitle emphasize * }
%      if$
%    }
%  if$
%}

%FUNCTION {empty.misc.check}
%{ author empty$ title empty$ howpublished empty$
%  month empty$ year empty$ note empty$
%  and and and and and
%    { "all relevant fields are empty in " cite$ * warning$ }
%    'skip$
%  if$
%}
%
%FUNCTION {format.thesis.type}
%{ type empty$
%    'skip$
%    { pop$
%      type "t" change.case$
%    }
%  if$
%}
%
%FUNCTION {format.tr.number}
%{ type empty$
%    { "Technical Report" }
%    'type
%  if$
%  number empty$
%    { "t" change.case$ }
%    { number tie.or.space.connect }
%  if$
%}
%
%FUNCTION {format.article.crossref}
%{ key empty$
%    { journal empty$
%        { "need key or journal for " cite$ * " to crossref " * crossref *
%          warning$
%          ""
%        }
%        { "In {\em " journal * "\/}" * }
%      if$
%    }
%    { "In " key * }
%  if$
%  " \cite{" * crossref * "}" *
%}
%
%FUNCTION {format.crossref.editor}
%{ editor #1 "{vv~}{ll}" format.name$
%  editor num.names$ duplicate$
%  #2 >
%    { pop$ " et~al." * }
%    { #2 <
%        'skip$
%        { editor #2 "{ff }{vv }{ll}{ jj}" format.name$ "others" =
%            { " et~al." * }
%            { " and " * editor #2 "{vv~}{ll}" format.name$ * }
%          if$
%        }
%      if$
%    }
%  if$
%}
%
%FUNCTION {format.book.crossref}
%{ volume empty$
%    { "empty volume in " cite$ * "'s crossref of " * crossref * warning$
%      "In "
%    }
%    { "Volume" volume tie.or.space.connect
%      " of " *
%    }
%  if$
%  editor empty$
%  editor field.or.null author field.or.null =
%  or
%    { key empty$
%        { series empty$
%            { "need editor, key, or series for " cite$ * " to crossref " *
%              crossref * warning$
%              "" *
%            }
%            { "{\em " * series * "\/}" * }
%          if$
%        }
%        { key * }
%      if$
%    }
%    { format.crossref.editor * }
%  if$
%  " \cite{" * crossref * "}" *
%}
%
%FUNCTION {format.incoll.inproc.crossref}
%{ editor empty$
%  editor field.or.null author field.or.null =
%  or
%    { key empty$
%        { booktitle empty$
%            { "need editor, key, or booktitle for " cite$ * " to crossref " *
%              crossref * warning$
%              ""
%            }
%            { "In {\em " booktitle * "\/}" * }
%          if$
%        }
%        { "In " key * }
%      if$
%    }
%    { "In " format.crossref.editor * }
%  if$
%  " \cite{" * crossref * "}" *
%}
%
%FUNCTION {article}
%{ output.bibitem
%  format.authors "author" output.check
%  new.block
%  format.title "title" output.check
%  new.block
%  crossref missing$
%    { journal emphasize "journal" output.check
%      format.vol.num.pages output
%      format.date "year" output.check
%    }
%    { format.article.crossref output.nonnull
%      format.pages output
%    }
%  if$
%  new.block
%  note output
%  fin.entry
%}
%
%FUNCTION {book}
%{ output.bibitem
%  author empty$
%    { format.editors "author and editor" output.check }
%    { format.authors output.nonnull
%      crossref missing$
%        { "author and editor" editor either.or.check }
%        'skip$
%      if$
%    }
%  if$
%  new.block
%  format.btitle "title" output.check
%  crossref missing$
%    { format.bvolume output
%      new.block
%      format.number.series output
%      new.sentence
%      publisher "publisher" output.check
%      address output
%    }
%    { new.block
%      format.book.crossref output.nonnull
%    }
%  if$
%  format.edition output
%  format.date "year" output.check
%  new.block
%  note output
%  fin.entry
%}
%
%FUNCTION {booklet}
%{ output.bibitem
%  format.authors output
%  new.block
%  format.title "title" output.check
%  howpublished address new.block.checkb
%  howpublished output
%  address output
%  format.date output
%  new.block
%  note output
%  fin.entry
%}
%
%FUNCTION {inbook}
%{ output.bibitem
%  author empty$
%    { format.editors "author and editor" output.check }
%    { format.authors output.nonnull
%      crossref missing$
%        { "author and editor" editor either.or.check }
%        'skip$
%      if$
%    }
%  if$
%  new.block
%  format.btitle "title" output.check
%  crossref missing$
%    { format.bvolume output
%      format.chapter.pages "chapter and pages" output.check
%      new.block
%      format.number.series output
%      new.sentence
%      publisher "publisher" output.check
%      address output
%    }
%    { format.chapter.pages "chapter and pages" output.check
%      new.block
%      format.book.crossref output.nonnull
%    }
%  if$
%  format.edition output
%  format.date "year" output.check
%  new.block
%  note output
%  fin.entry
%}
%
%FUNCTION {incollection}
%{ output.bibitem
%  format.authors "author" output.check
%  new.block
%  format.title "title" output.check
%  new.block
%  crossref missing$
%    { format.in.ed.booktitle "booktitle" output.check
%      format.bvolume output
%      format.number.series output
%      format.chapter.pages output
%      new.sentence
%      publisher "publisher" output.check
%      address output
%      format.edition output
%      format.date "year" output.check
%    }
%    { format.incoll.inproc.crossref output.nonnull
%      format.chapter.pages output
%    }
%  if$
%  new.block
%  note output
%  fin.entry
%}
%
%FUNCTION {inproceedings}
%{ output.bibitem
%  format.authors "author" output.check
%  new.block
%  format.title "title" output.check
%  new.block
%  crossref missing$
%    { format.in.ed.booktitle "booktitle" output.check
%      format.bvolume output
%      format.number.series output
%      format.pages output
%      address empty$
%        { organization publisher new.sentence.checkb
%          organization output
%          publisher output
%          format.date "year" output.check
%        }
%        { address output.nonnull
%          format.date "year" output.check
%          new.sentence
%          organization output
%          publisher output
%        }
%      if$
%    }
%    { format.incoll.inproc.crossref output.nonnull
%      format.pages output
%    }
%  if$
%  new.block
%  note output
%  fin.entry
%}
%
%FUNCTION {conference} { inproceedings }
%
%FUNCTION {manual}
%{ output.bibitem
%  author empty$
%    { organization empty$
%        'skip$
%        { organization output.nonnull
%          address output
%        }
%      if$
%    }
%    { format.authors output.nonnull }
%  if$
%  new.block
%  format.btitle "title" output.check
%  author empty$
%    { organization empty$
%        { address new.block.checka
%          address output
%        }
%        'skip$
%      if$
%    }
%    { organization address new.block.checkb
%      organization output
%      address output
%    }
%  if$
%  format.edition output
%  format.date output
%  new.block
%  note output
%  fin.entry
%}
%
%FUNCTION {mastersthesis}
%{ output.bibitem
%  format.authors "author" output.check
%  new.block
%  format.title "title" output.check
%  new.block
%  "Master's thesis" format.thesis.type output.nonnull
%  school "school" output.check
%  address output
%  format.date "year" output.check
%  new.block
%  note output
%  fin.entry
%}
%
%FUNCTION {misc}
%{ output.bibitem
%  format.authors output
%  title howpublished new.block.checkb
%  format.title output
%  howpublished new.block.checka
%  howpublished output
%  format.date output
%  new.block
%  note output
%  fin.entry
%  empty.misc.check
%}
%
%FUNCTION {phdthesis}
%{ output.bibitem
%  format.authors "author" output.check
%  new.block
%  format.btitle "title" output.check
%  new.block
%  "PhD thesis" format.thesis.type output.nonnull
%  school "school" output.check
%  address output
%  format.date "year" output.check
%  new.block
%  note output
%  fin.entry
%}
%
%FUNCTION {proceedings}
%{ output.bibitem
%  editor empty$
%    { organization output }
%    { format.editors output.nonnull }
%  if$
%  new.block
%  format.btitle "title" output.check
%  format.bvolume output
%  format.number.series output
%  address empty$
%    { editor empty$
%        { publisher new.sentence.checka }
%        { organization publisher new.sentence.checkb
%          organization output
%        }
%      if$
%      publisher output
%      format.date "year" output.check
%    }
%    { address output.nonnull
%      format.date "year" output.check
%      new.sentence
%      editor empty$
%        'skip$
%        { organization output }
%      if$
%      publisher output
%    }
%  if$
%  new.block
%  note output
%  fin.entry
%}
%
%FUNCTION {techreport}
%{ output.bibitem
%  format.authors "author" output.check
%  new.block
%  format.title "title" output.check
%  new.block
%  format.tr.number output.nonnull
%  institution "institution" output.check
%  address output
%  format.date "year" output.check
%  new.block
%  note output
%  fin.entry
%}
%
%FUNCTION {unpublished}
%{ output.bibitem
%  format.authors "author" output.check
%  new.block
%  format.title "title" output.check
%  new.block
%  note "note" output.check
%  format.date output
%  fin.entry
%}

FUNCTION {empty.misc.check}
{ author empty$ title empty$ howpublished empty$
  month empty$ year empty$ note empty$
  and and and and and
  key empty$ not and
    { "all relevant fields are empty in " cite$ * warning$ }
    'skip$
  if$
}
FUNCTION {format.thesis.type}
{ type duplicate$ empty$
    'pop$
    { swap$ pop$
      "t" change.case$ "type" bibinfo.check
    }
  if$
}
FUNCTION {format.tr.number}
{ number "number" bibinfo.check
  type duplicate$ empty$
    { pop$ bbl.techrep }
    'skip$
  if$
  "type" bibinfo.check
  swap$ duplicate$ empty$
    { pop$ "t" change.case$ }
    { tie.or.space.prefix * * }
  if$
}
FUNCTION {format.article.crossref}
{
  key duplicate$ empty$
    { pop$
      journal duplicate$ empty$
        { "need key or journal for " cite$ * " to crossref " * crossref * warning$ }
        { "journal" bibinfo.check emphasize word.in swap$ * }
      if$
    }
    { word.in swap$ * " " *}
  if$
  " \cite{" * crossref * "}" *
}
FUNCTION {format.crossref.editor}
{ editor #1 "{vv~}{ll}" format.name$
  "editor" bibinfo.check
  editor num.names$ duplicate$
  #2 >
    { pop$
      "editor" bibinfo.check
      " " * bbl.etal
      *
    }
    { #2 <
        'skip$
        { editor #2 "{ff }{vv }{ll}{ jj}" format.name$ "others" =
            {
              "editor" bibinfo.check
              " " * bbl.etal
              *
            }
            {
             bbl.and space.word
              * editor #2 "{vv~}{ll}" format.name$
              "editor" bibinfo.check
              *
            }
          if$
        }
      if$
    }
  if$
}
FUNCTION {format.book.crossref}
{ volume duplicate$ empty$
    { "empty volume in " cite$ * "'s crossref of " * crossref * warning$
      pop$ word.in
    }
    { bbl.volume
      capitalize
      swap$ tie.or.space.prefix "volume" bibinfo.check * * bbl.of space.word *
    }
  if$
  editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
        { series empty$
            { "need editor, key, or series for " cite$ * " to crossref " *
              crossref * warning$
              "" *
            }
            { series emphasize * }
          if$
        }
        { key * }
      if$
    }
    { format.crossref.editor * }
  if$
  " \cite{" * crossref * "}" *
}
FUNCTION {format.incoll.inproc.crossref}
{
  editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
        { format.booktitle duplicate$ empty$
            { "need editor, key, or booktitle for " cite$ * " to crossref " *
              crossref * warning$
            }
            { word.in swap$ * }
          if$
        }
        { word.in key * " " *}
      if$
    }
    { word.in format.crossref.editor * " " *}
  if$
  " \cite{" * crossref * "}" *
}
FUNCTION {format.org.or.pub}
{ 't :=
  ""
  address empty$ t empty$ and
    'skip$
    {
      t empty$
        { address "address" bibinfo.check *
        }
        { t *
          address empty$
            'skip$
            { ", " * address "address" bibinfo.check * }
          if$
        }
      if$
    }
  if$
}
FUNCTION {format.publisher.address}
{ publisher "publisher" bibinfo.warn format.org.or.pub
}

FUNCTION {format.organization.address}
{ organization "organization" bibinfo.check format.org.or.pub
}

FUNCTION {article}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.title "title" output.check
  new.block
  crossref missing$
    {
      journal
      "journal" bibinfo.check
      emphasize
      "journal" output.check
      format.vol.num.pages output
    }
    { format.article.crossref output.nonnull
      format.pages output
    }
  if$
  new.block
  format.url output
  new.block
  format.note output
  fin.entry
}

FUNCTION {book}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check
    }
    { format.authors output.nonnull
      crossref missing$
        { "author and editor" editor either.or.check }
        'skip$
      if$
    }
  if$
  new.block
  format.btitle "title" output.check
  crossref missing$
    { format.bvolume output
      new.block
      format.edition output
      new.block
%      new.sentence
      format.number.series output
      format.publisher.address output
    }
    {
      new.block
      format.book.crossref output.nonnull
    }
  if$
  format.date "year" output.check
  new.block
  format.url output
  new.block
  format.note output
  fin.entry
}

FUNCTION {booklet}
{ output.bibitem
  format.authors output
  new.block
  format.title "title" output.check
  new.block
  howpublished "howpublished" bibinfo.check output
  address "address" bibinfo.check output
  format.date output
  new.block
  format.url output
  new.block
  format.note output
  fin.entry
}

FUNCTION {inbook}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check
    }
    { format.authors output.nonnull
      crossref missing$
        { "author and editor" editor either.or.check }
        'skip$
      if$
    }
  if$
  new.block
  format.btitle "title" output.check
  crossref missing$
    {
      format.bvolume output
      format.chapter.pages "chapter and pages" output.check
      new.block
      format.edition output
      new.sentence
      format.number.series output
      format.publisher.address output
    }
    {
      format.chapter.pages "chapter and pages" output.check
      new.block
      format.book.crossref output.nonnull
    }
  if$
  format.date "year" output.check
  new.block
  format.url output
  new.block
  format.note output
  fin.entry
}

FUNCTION {incollection}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.title "title" output.check
  new.block
  crossref missing$
    { format.in.ed.booktitle "booktitle" output.check
      format.bvolume output
      format.edition output
%      format.chapter.pages output
      new.sentence
      format.number.series output
      format.publisher.address output
    }
    { format.incoll.inproc.crossref output.nonnull
      format.chapter.pages output
    }
  if$
  format.date "year" output.check
  new.block
  format.chapter.pages output
  new.block
  format.url output
  new.block
  format.note output
  fin.entry
}

FUNCTION {inproceedings}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.title "title" output.check
  new.block
  crossref missing$
    { format.in.ed.booktitle "booktitle" output.check
      format.bvolume output
      new.sentence
      format.number.series output
      publisher empty$
        { format.organization.address output }
        { organization "organization" bibinfo.check output
          format.publisher.address output
        }
      if$
    }
    { format.incoll.inproc.crossref output.nonnull
    }
  if$
  format.date "year" output.check
  new.block
  format.pages "pages" output.check
  new.block
  format.url output
  new.block
  format.note output
  fin.entry
}

FUNCTION {conference} { inproceedings }

FUNCTION {manual}
{ output.bibitem
  author empty$
    { organization "organization" bibinfo.check
      duplicate$ empty$ 'pop$
        { output
          address "address" bibinfo.check output
        }
      if$
    }
    { format.authors output.nonnull }
  if$
  new.block
  format.btitle "title" output.check
  format.edition output
  author empty$
    { organization empty$
        {
          address new.block.checka
          address "address" bibinfo.check output
        }
        'skip$
      if$
    }
    {
      organization address new.block.checkb
      organization "organization" bibinfo.check output
      address "address" bibinfo.check output
    }
  if$
%  format.edition output
  format.date output
  new.block
  format.url output
  new.block
  format.note output
  fin.entry
}

FUNCTION {mastersthesis}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.btitle
  "title" output.check
  new.block
  bbl.mthesis format.thesis.type output.nonnull
  school "school" bibinfo.warn output
  address "address" bibinfo.check output
  format.date "year" output.check
  new.block
  format.url output
  new.block
  format.note output
  fin.entry
}

FUNCTION {misc}
{ output.bibitem
  format.authors output
  title howpublished new.block.checkb
  format.title output
  howpublished new.block.checka
  howpublished "howpublished" bibinfo.check output
  format.date output
  new.block
  format.url output
  new.block
  format.note output
  fin.entry
  empty.misc.check
}

FUNCTION {phdthesis}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.btitle
  "title" output.check
  new.block
  bbl.phdthesis format.thesis.type output.nonnull
  school "school" bibinfo.warn output
  address "address" bibinfo.check output
  format.date "year" output.check
  new.block
  format.url output
  new.block
  format.note output
  fin.entry
}

FUNCTION {proceedings}
{ output.bibitem
  editor empty$
    { organization "organization" bibinfo.check output
    }
    { format.editors output.nonnull }
  if$
  new.block
  format.btitle "title" output.check
  format.bvolume output
  editor empty$
    { publisher empty$
        {  format.number.series output }
        {
          new.sentence
          format.number.series output
          format.publisher.address output
        }
      if$
    }
    { publisher empty$
        {
          new.sentence
          format.number.series output
          format.organization.address output }
        {
          new.sentence
          format.number.series output
          organization "organization" bibinfo.check output
          format.publisher.address output
        }
      if$
     }
  if$
      format.date "year" output.check
  new.block
  format.url output
  new.block
  format.note output
  fin.entry
}

FUNCTION {techreport}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.btitle
  "title" output.check
  new.block
  format.tr.number output.nonnull
  institution "institution" bibinfo.warn output
  address "address" bibinfo.check output
  format.date "year" output.check
  new.block
  format.url output
  new.block
  format.note output
  fin.entry
}

FUNCTION {unpublished}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.title "title" output.check
  format.date output
  new.block
  format.arxiv output
  new.block
  format.url output
  new.block
  format.note "note" output.check
  fin.entry
}

FUNCTION {default.type} { misc }

MACRO {jan} {"January"}

MACRO {feb} {"February"}

MACRO {mar} {"March"}

MACRO {apr} {"April"}

MACRO {may} {"May"}

MACRO {jun} {"June"}

MACRO {jul} {"July"}

MACRO {aug} {"August"}

MACRO {sep} {"September"}

MACRO {oct} {"October"}

MACRO {nov} {"November"}

MACRO {dec} {"December"}

MACRO {acmcs} {"ACM Computing Surveys"}

MACRO {acta} {"Acta Informatica"}

MACRO {cacm} {"Communications of the ACM"}

MACRO {ibmjrd} {"IBM Journal of Research and Development"}

MACRO {ibmsj} {"IBM Systems Journal"}

MACRO {ieeese} {"IEEE Transactions on Software Engineering"}

MACRO {ieeetc} {"IEEE Transactions on Computers"}

MACRO {ieeetcad}
 {"IEEE Transactions on Computer-Aided Design of Integrated Circuits"}

MACRO {ipl} {"Information Processing Letters"}

MACRO {jacm} {"Journal of the ACM"}

MACRO {jcss} {"Journal of Computer and System Sciences"}

MACRO {scp} {"Science of Computer Programming"}

MACRO {sicomp} {"SIAM Journal on Computing"}

MACRO {tocs} {"ACM Transactions on Computer Systems"}

MACRO {tods} {"ACM Transactions on Database Systems"}

MACRO {tog} {"ACM Transactions on Graphics"}

MACRO {toms} {"ACM Transactions on Mathematical Software"}

MACRO {toois} {"ACM Transactions on Office Information Systems"}

MACRO {toplas} {"ACM Transactions on Programming Languages and Systems"}

MACRO {tcs} {"Theoretical Computer Science"}

READ

STRINGS { longest.label }

INTEGERS { number.label longest.label.width }

FUNCTION {initialize.longest.label}
{ "" 'longest.label :=
  #1 'number.label :=
  #0 'longest.label.width :=
}

FUNCTION {longest.label.pass}
{ number.label int.to.str$ 'label :=
  number.label #1 + 'number.label :=
  label width$ longest.label.width >
    { label 'longest.label :=
      label width$ 'longest.label.width :=
    }
    'skip$
  if$
}

EXECUTE {initialize.longest.label}

ITERATE {longest.label.pass}

FUNCTION {begin.bib}
{ preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  "\providecommand{\href}[2]{#2}"
  write$ newline$
  "\providecommand{\arxiv}[1]{\href{http://arxiv.org/abs/#1}{arXiv:#1}}"
  write$ newline$
  "\providecommand{\url}[1]{\texttt{#1}}"
  write$ newline$
  "\providecommand{\urlprefix}{URL }"
  write$ newline$
  "\begin{thebibliography}{"  longest.label  * "}" *
  write$ newline$
}

%FUNCTION {begin.bib}
%{ preamble$ empty$
%    'skip$
%    { preamble$ write$ newline$ }
%  if$
%  "\begin{thebibliography}{"  longest.label  * "}" * write$ newline$
%}

EXECUTE {begin.bib}

EXECUTE {init.state.consts}

ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
}

EXECUTE {end.bib}
