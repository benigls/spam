#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    'test_dataset',
)

DATASET_SUBDIRS = [
    {
        'name': 'enron1',
        'total_count': 5,
        'ham_count': 3,
        'spam_count': 2,
        'path': os.path.join(DATASET_PATH, 'enron1'),
        'ham_path': os.path.join(
            DATASET_PATH,
            'enron1',
            'ham'
        ),
        'spam_path': os.path.join(
            DATASET_PATH,
            'enron1',
            'spam'
        ),
        'ham_emails': [
            # 0019.1999-12-15.farmer.ham.txt
            """
            Subject: meter 1431 - nov 1999
            aimee ,
            sitara deal 92943 for meter 1431 has expired on oct 31 ,
            1999 . settlements
            is unable to draft an invoice for this deal .
            this deal either needs to be
            extended or a new deal needs to be set up .
            please let me know when this is
            resolved . we need it resolved by friday , dec 17 .
            hc
            """,
            # 0028.1999-12-17.farmer.ham.txt
            """
            Subject: pennzenergy property details

            - - - - - - - - - - - - - - - - - - - - - - forwarded
            by ami chokshi / corp / enron on 12 / 17 / 99 04 : 03

            pm - - - - - - - - - - - - - - - - - - - - - - - - - - -
            dscottl @ . com on 12 / 14 / 99 10 : 56 : 01 am
            to : ami chokshi / corp / enron @ enron
            cc :
            subject : pennzenergy property details
            ami , attached is some more details on the devon south
            texas properties . let
            me
            know if you have any questions .
            david
            - devon stx . xls
            """,
            # 5149.2002-01-04.farmer.ham.txt
            """
            Subject: contract status needed
            on monday we have to file a " transition plan " with
            louise kitchen for netco . she wants details . on that
            note i need a status report from each desk on our
            effort to start the contract process for pipes , tpa
            ' s & ebb ' s so i can include that in the report .
            i have received some emails however i would like each
            manager to put together a summary in an excel
            spreadsheet that we can have suzanne consolidate with
            the stuff she already put together . please cover the
            key items listed in the earlier email . also we should
            be getting as much hard copy as possible so we can
            proceed with our goal of preparing for a startup . i
            suggest reviewing the paperwork and identify the stuff
            that you will need to get the forms completed . no
            doubt we will have substantial gaps but as long as we
            know where and what they are we can get the proper
            internal people involved ( legal , credit , etc . )
            tks - bob
            """,
        ],
        'spam_emails': [
            """
            """,
            """
            """,
            """
            """,
        ],
    },
    {
        'name': 'enron2',
        'total_count': 6,
        'ham_count': 2,
        'spam_count': 4,
        'path': os.path.join(DATASET_PATH, 'enron2'),
        'ham_path': os.path.join(
            DATASET_PATH,
            'enron2',
            'ham'
        ),
        'spam_path': os.path.join(
            DATASET_PATH,
            'enron2',
            'spam'
        ),
        'ham_emails': [
            """
            """,
            """
            """,
            """
            """,
        ],
        'spam_emails': [
            """
            """,
            """
            """,
            """
            """,
        ],
    },
]
