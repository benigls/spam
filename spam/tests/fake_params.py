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
            # 0011.2003-12-18.GP.spam.txt
            """
            Subject: sup . er cha . rge your m . an hood today jvbe kfbtyra xes
            hello ,
            generic and super viagra ( cialis ) available online !
            most trusted online source !
            cialis or ( super viag )
            takes affect right away & lasts 24 - 36 hours !
            for super viagra click here
            generic viagra
            costs 60 % less ! save a lot of money .
            for viagra click here
            both products shipped discretely to your door
            not interested ?
            dycmpf
            s uuz
            biwven
            """,
            # 0030.2003-12-21.GP.spam.txt
            """
            Subject: medical info :
            health watch for 2003
            four seconds until picture is downloaded
            if you hadn ' t happened to find the piglet , eureka
            would surely have been executedyour body can not be
            very valuable to you if all your time is required to
            feed it'
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
            # 0028.2001-02-15.kitchen.ham.txt
            """
            Subject: elektro short position
            for this morning call thanks
            - - - - - - - - - - - - - - - - - - - - - - forwarded by
            orlando gonzalez / sa / enron on 15 / 02 / 2001 12 : 14 -
            - - - - - - - - - - - - - - - - - - - - - - - - - -
            luis henriques
            15 / 02 / 2001 12 : 06
            to : orlando gonzalez / sa / enron @ enron
            cc :
            subject : elektro short position
            see attached a file for our today conference call .
            luiz otavio
            """,
            # 0042.2001-02-20.kitchen.ham.txt
            """
            Subject: draft press release : kcs energy
            this press release for kcs energy has a reference to
            ena . call me if you have any problems with the
            attached . it is scheduled to go out today .
            eric
            - press release 2 - 20 - 01 final . doc
            """,
        ],
        'spam_emails': [
            # 0037.2003-12-22.GP.spam.txt
            """
            Subject: re [ 8 ] : dear friend -
            size = 1 > order confirmation . your order should be
            shipped by january , via fedex .
            your federal express tracking number is 45954036 .
            thank you for registering . your userid is : 56075519
            learn to make a fortune with ebay !
            complete turnkey system software - videos - turorials
            clk here for information
            clilings .
            """,
            # 0107.2004-08-11.BG.spam.txt
            """
            Subject: make $ 314
            hello ,
            we sent you an email a while ago , because you now
            qualify for a new mortgage .
            you could get $ 300 , 000 for as little as $ 700 a
            month !
            bad credit is no problem , you can pull cash out or
            refinance .
            please click on this link for free consultation by a
            mortgage broker :
            http : / / www . hgkkdc . com /
            best regards ,
            jamie higgs
            no thanks : http : / / www . hgkkdc . com / rl
            - - - - system information - - - -
            form from area works numbering another contribution zone
            languages ) rfc 3066 languages essential require
            presentation submitted cannot
        stroke - radical here collating resources intermediaries
        identified presents
        index
            """,
            # 0131.2004-08-12.BG.spam.txt
            """
            Subject: the p : ri , ce is r ; ight
            look at the crazy prices of these high - end software packages !
            why buy them for hundreds of dollars at the store ?
            try them out here , they ' re 100 % money back guarantee .
            windows xp professional 2002
            . . . . . . . . . . . . . $ 50
            adobe photoshop 7 . 0
            . . . . . . . . . . . . . . . . . . . . . $ 60
            microsoft office xp professional 2002
            . . . $ 60
            corel draw graphics suite 11
            . . . . . . . . . . . . $ 60
            and lots more . . .'
            """,
            # 0044.2004-08-06.BG.spam.txt
            """
            Subject: give your partner more pleasure
            my girlfriend loves the results , but she doesn ' t know
            what i do . she
            thinks
            it ' s natural - thomas , ca
            i ' ve been using your product for 4 months now . i ' ve
            increased my
            length from 2
            to nearly 6 . your product has saved my sex life .
            - matt , fl
            pleasure your partner every time with a bigger , longer ,
            stronger unit
            realistic gains quickly
            to be a stud
            press here
            and now i must go , he continued , for my stay in your
            city will be a
            short one and i want to see all i can
            oranjestad ,
            aruba , po b 1200
            come , ozma , she said , anxiously ; let us go ourselves
            to search for the
            piglet the president scrawled something on a sheet of
            paper and signed his
            name to it , afterward presenting it , with a courteous
            bow , to his visitor
            """,
        ],
    },
]
