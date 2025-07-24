# Dataset Inspection Report
This report details the structure of the dataset used in the project.

## 1. Dataset Features (Column Names and Data Types)
This section shows the schema of the dataset, including each column's name and its expected data type.
```
{'pre_text': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None), 'post_text': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None), 'filename': Value(dtype='string', id=None), 'table_ori': Sequence(feature=Sequence(feature=Value(dtype='string', id=None), length=-1, id=None), length=-1, id=None), 'table': Sequence(feature=Sequence(feature=Value(dtype='string', id=None), length=-1, id=None), length=-1, id=None), 'qa': {'ann_table_rows': Sequence(feature=Value(dtype='null', id=None), length=-1, id=None), 'ann_text_rows': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), 'answer': Value(dtype='string', id=None), 'exe_ans': Value(dtype='string', id=None), 'explanation': Value(dtype='string', id=None), 'gold_inds': {'text_1': Value(dtype='string', id=None)}, 'model_input': Sequence(feature=Sequence(feature=Value(dtype='string', id=None), length=-1, id=None), length=-1, id=None), 'program': Value(dtype='string', id=None), 'program_re': Value(dtype='string', id=None), 'question': Value(dtype='string', id=None), 'steps': [{'arg1': Value(dtype='string', id=None), 'arg2': Value(dtype='string', id=None), 'op': Value(dtype='string', id=None), 'res': Value(dtype='string', id=None)}], 'tfidftopn': {'text_0': Value(dtype='string', id=None), 'text_14': Value(dtype='string', id=None)}}, 'id': Value(dtype='string', id=None), 'table_retrieved': [{'ind': Value(dtype='string', id=None), 'score': Value(dtype='float64', id=None)}], 'text_retrieved': [{'ind': Value(dtype='string', id=None), 'score': Value(dtype='float64', id=None)}], 'table_retrieved_all': [{'ind': Value(dtype='string', id=None), 'score': Value(dtype='float64', id=None)}], 'text_retrieved_all': [{'ind': Value(dtype='string', id=None), 'score': Value(dtype='float64', id=None)}]}
```

## 2. Column Names (as a list)
A simple list of all available column names. **Use these exact names in your code.**
```
['pre_text', 'post_text', 'filename', 'table_ori', 'table', 'qa', 'id', 'table_retrieved', 'text_retrieved', 'table_retrieved_all', 'text_retrieved_all']
```

## 3. First Example Record
Below is the full data for the first record in the training set. This helps visualize the content of each column.
```python
{ 'filename': 'ADI/2009/page_49.pdf',
  'id': 'ADI/2009/page_49.pdf-1',
  'post_text': [ 'fair value of forward exchange contracts after a 10% ( 10 % '
                 ') unfavorable movement in foreign currency exchange rates '
                 'asset ( liability ) .',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '$ 20132 $ ( 9457 ) fair value of forward exchange contracts '
                 'after a 10% ( 10 % ) favorable movement in foreign currency '
                 'exchange rates liability .',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '.',
                 '$ ( 6781 ) $ ( 38294 ) the calculation assumes that each '
                 'exchange rate would change in the same direction relative to '
                 'the u.s .',
                 'dollar .',
                 'in addition to the direct effects of changes in exchange '
                 'rates , such changes typically affect the volume of sales or '
                 'the foreign currency sales price as competitors 2019 '
                 'products become more or less attractive .',
                 'our sensitivity analysis of the effects of changes in '
                 'foreign currency exchange rates does not factor in a '
                 'potential change in sales levels or local currency selling '
                 'prices. .'],
  'pre_text': [ 'interest rate to a variable interest rate based on the '
                'three-month libor plus 2.05% ( 2.05 % ) ( 2.34% ( 2.34 % ) as '
                'of october 31 , 2009 ) .',
                'if libor changes by 100 basis points , our annual interest '
                'expense would change by $ 3.8 million .',
                'foreign currency exposure as more fully described in note 2i '
                '.',
                'in the notes to consolidated financial statements contained '
                'in item 8 of this annual report on form 10-k , we regularly '
                'hedge our non-u.s .',
                'dollar-based exposures by entering into forward foreign '
                'currency exchange contracts .',
                'the terms of these contracts are for periods matching the '
                'duration of the underlying exposure and generally range from '
                'one month to twelve months .',
                'currently , our largest foreign currency exposure is the euro '
                ', primarily because our european operations have the highest '
                'proportion of our local currency denominated expenses .',
                'relative to foreign currency exposures existing at october 31 '
                ', 2009 and november 1 , 2008 , a 10% ( 10 % ) unfavorable '
                'movement in foreign currency exchange rates over the course '
                'of the year would not expose us to significant losses in '
                'earnings or cash flows because we hedge a high proportion of '
                'our year-end exposures against fluctuations in foreign '
                'currency exchange rates .',
                'the market risk associated with our derivative instruments '
                'results from currency exchange rate or interest rate '
                'movements that are expected to offset the market risk of the '
                'underlying transactions , assets and liabilities being hedged '
                '.',
                'the counterparties to the agreements relating to our foreign '
                'exchange instruments consist of a number of major '
                'international financial institutions with high credit ratings '
                '.',
                'we do not believe that there is significant risk of '
                'nonperformance by these counterparties because we continually '
                'monitor the credit ratings of such counterparties .',
                'while the contract or notional amounts of derivative '
                'financial instruments provide one measure of the volume of '
                'these transactions , they do not represent the amount of our '
                'exposure to credit risk .',
                'the amounts potentially subject to credit risk ( arising from '
                'the possible inability of counterparties to meet the terms of '
                'their contracts ) are generally limited to the amounts , if '
                'any , by which the counterparties 2019 obligations under the '
                'contracts exceed our obligations to the counterparties .',
                'the following table illustrates the effect that a 10% ( 10 % '
                ') unfavorable or favorable movement in foreign currency '
                'exchange rates , relative to the u.s .',
                'dollar , would have on the fair value of our forward exchange '
                'contracts as of october 31 , 2009 and november 1 , 2008: .'],
  'qa': { 'ann_table_rows': [],
          'ann_text_rows': [1],
          'answer': '380',
          'exe_ans': '3.8',
          'explanation': '',
          'gold_inds': { 'text_1': 'if libor changes by 100 basis points , our '
                                   'annual interest expense would change by $ '
                                   '3.8 million .'},
          'model_input': [ [ 'text_0',
                             'interest rate to a variable interest rate based '
                             'on the three-month libor plus 2.05% ( 2.05 % ) ( '
                             '2.34% ( 2.34 % ) as of october 31 , 2009 ) .'],
                           [ 'text_1',
                             'if libor changes by 100 basis points , our '
                             'annual interest expense would change by $ 3.8 '
                             'million .'],
                           [ 'text_14',
                             'dollar , would have on the fair value of our '
                             'forward exchange contracts as of october 31 , '
                             '2009 and november 1 , 2008: .']],
          'program': 'divide(100, 100), divide(3.8, #0)',
          'program_re': 'divide(3.8, divide(100, 100))',
          'question': 'what is the the interest expense in 2009?',
          'steps': [ { 'arg1': '100',
                       'arg2': '100',
                       'op': 'divide1-1',
                       'res': '1%'},
                     { 'arg1': '3.8',
                       'arg2': '#0',
                       'op': 'divide1-2',
                       'res': '380'}],
          'tfidftopn': { 'text_0': 'interest rate to a variable interest rate '
                                   'based on the three-month libor plus 2.05% '
                                   '( 2.05 % ) ( 2.34% ( 2.34 % ) as of '
                                   'october 31 , 2009 ) .',
                         'text_14': 'dollar , would have on the fair value of '
                                    'our forward exchange contracts as of '
                                    'october 31 , 2009 and november 1 , 2008: '
                                    '.'}},
  'table': [ ['', 'october 31 2009', 'november 1 2008'],
             [ 'fair value of forward exchange contracts asset ( liability )',
               '$ 6427',
               '$ -23158 ( 23158 )'],
             [ 'fair value of forward exchange contracts after a 10% ( 10 % ) '
               'unfavorable movement in foreign currency exchange rates asset '
               '( liability )',
               '$ 20132',
               '$ -9457 ( 9457 )'],
             [ 'fair value of forward exchange contracts after a 10% ( 10 % ) '
               'favorable movement in foreign currency exchange rates '
               'liability',
               '$ -6781 ( 6781 )',
               '$ -38294 ( 38294 )']],
  'table_ori': [ ['', 'October 31, 2009', 'November 1, 2008'],
                 [ 'Fair value of forward exchange contracts asset (liability)',
                   '$6,427',
                   '$(23,158)'],
                 [ 'Fair value of forward exchange contracts after a 10% '
                   'unfavorable movement in foreign currency exchange rates '
                   'asset (liability)',
                   '$20,132',
                   '$(9,457)'],
                 [ 'Fair value of forward exchange contracts after a 10% '
                   'favorable movement in foreign currency exchange rates '
                   'liability',
                   '$(6,781)',
                   '$(38,294)']],
  'table_retrieved': [ {'ind': 'table_1', 'score': -0.6207679510116577},
                       {'ind': 'table_2', 'score': -0.8948984742164612}],
  'table_retrieved_all': [ {'ind': 'table_1', 'score': -0.6207679510116577},
                           {'ind': 'table_2', 'score': -0.8948984742164612},
                           {'ind': 'table_3', 'score': -1.2129878997802734},
                           {'ind': 'table_0', 'score': -2.9782934188842773}],
  'text_retrieved': [ {'ind': 'text_1', 'score': 1.251369595527649},
                      {'ind': 'text_0', 'score': 0.6589734554290771},
                      {'ind': 'text_14', 'score': -0.1914736032485962}],
  'text_retrieved_all': [ {'ind': 'text_1', 'score': 1.251369595527649},
                          {'ind': 'text_0', 'score': 0.6589734554290771},
                          {'ind': 'text_14', 'score': -0.1914736032485962},
                          {'ind': 'text_47', 'score': -1.0945320129394531},
                          {'ind': 'text_24', 'score': -1.4916260242462158},
                          {'ind': 'text_12', 'score': -1.5615578889846802},
                          {'ind': 'text_15', 'score': -1.572263479232788},
                          {'ind': 'text_5', 'score': -1.6337369680404663},
                          {'ind': 'text_3', 'score': -1.678298830986023},
                          {'ind': 'text_6', 'score': -1.6905218362808228},
                          {'ind': 'text_46', 'score': -1.9114893674850464},
                          {'ind': 'text_7', 'score': -1.914547324180603},
                          {'ind': 'text_8', 'score': -1.955027461051941},
                          {'ind': 'text_49', 'score': -2.0304598808288574},
                          {'ind': 'text_13', 'score': -2.0383174419403076},
                          {'ind': 'text_10', 'score': -2.112241268157959},
                          {'ind': 'text_11', 'score': -2.1439552307128906},
                          {'ind': 'text_4', 'score': -2.2258567810058594},
                          {'ind': 'text_48', 'score': -2.409395694732666},
                          {'ind': 'text_9', 'score': -2.6092159748077393},
                          {'ind': 'text_2', 'score': -2.6313436031341553},
                          {'ind': 'text_16', 'score': -2.932347536087036},
                          {'ind': 'text_17', 'score': -2.932347536087036},
                          {'ind': 'text_18', 'score': -2.932347536087036},
                          {'ind': 'text_19', 'score': -2.932347536087036},
                          {'ind': 'text_20', 'score': -2.932347536087036},
                          {'ind': 'text_21', 'score': -2.932347536087036},
                          {'ind': 'text_22', 'score': -2.932347536087036},
                          {'ind': 'text_23', 'score': -2.932347536087036},
                          {'ind': 'text_25', 'score': -2.932347536087036},
                          {'ind': 'text_26', 'score': -2.932347536087036},
                          {'ind': 'text_27', 'score': -2.932347536087036},
                          {'ind': 'text_28', 'score': -2.932347536087036},
                          {'ind': 'text_29', 'score': -2.932347536087036},
                          {'ind': 'text_30', 'score': -2.932347536087036},
                          {'ind': 'text_31', 'score': -2.932347536087036},
                          {'ind': 'text_32', 'score': -2.932347536087036},
                          {'ind': 'text_33', 'score': -2.932347536087036},
                          {'ind': 'text_34', 'score': -2.932347536087036},
                          {'ind': 'text_35', 'score': -2.932347536087036},
                          {'ind': 'text_36', 'score': -2.932347536087036},
                          {'ind': 'text_37', 'score': -2.932347536087036},
                          {'ind': 'text_38', 'score': -2.932347536087036},
                          {'ind': 'text_39', 'score': -2.932347536087036},
                          {'ind': 'text_40', 'score': -2.932347536087036},
                          {'ind': 'text_41', 'score': -2.932347536087036},
                          {'ind': 'text_42', 'score': -2.932347536087036},
                          {'ind': 'text_43', 'score': -2.932347536087036},
                          {'ind': 'text_44', 'score': -2.932347536087036},
                          {'ind': 'text_45', 'score': -2.932347536087036}]}
```