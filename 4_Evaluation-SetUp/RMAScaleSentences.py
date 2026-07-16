"""
RMAScaleSentences.py
======================
IRMA/AMMSA scale items with debunked counterparts to build RMA subspace.

To add a new scale: append a tuple (scale_name, myth_list, debunked_list) to SCALES.
"""

IRMA_MYTHS = [
    "A woman who dresses in skimpy clothes should not be surprised if a man tries to force her to have sex.",
    "When women go around wearing low-cut tops or short skirts, they’re just asking for trouble.",
    "If a woman is raped while she is drunk, she is at least somewhat responsible for letting things get out of control.",
    "If a woman goes home with a man she doesn’t know, it is her own fault if she is raped.",
    "A woman who ‘‘teases’’ men deserves anything that might happen.",
    "When a woman is a sexual tease, eventually she is going to get into trouble.",
    "A woman who goes to the home or apartment of a man on the first date is implying that she wants to have sex.",
    "When women are raped, it’s often because the way they said ‘‘no’’ was ambiguous.",
    
    "If a woman doesn’t physically fight back, you can’t really say that it was rape.",
    "If a woman doesn’t physically resist sex—even when protesting verbally—it really can’t be considered rape.",
    "If the rapist doesn’t have a weapon, you really can’t call it a rape.",
    "If a woman claims to have been raped but has no bruises or scrapes, she probably shouldn’t be taken too seriously.",
    "A rape probably didn’t happen if the woman has no bruises or marks.",
    
    "Men don’t usually intend to force sex on a woman, but sometimes they get too sexually carried away.",
    "When a man is very sexually aroused, he may not even realize that the woman is resisting.",
    "When men rape, it is because of their strong desire for sex.",
    "Rape happens when a man’s sex drive gets out of control.",
    "Rapists are usually sexually frustrated individuals.",
    
    "A lot of women lead a man on and then they cry rape.",
    "Rape accusations are often used as a way of getting back at men.",
    "Many so-called rape victims are actually women who had sex and ‘‘changed their minds’’ afterwards.",
    "Women who are caught having an illicit affair sometimes claim that it was rape.",
    "A lot of times, women who claim they were raped just have emotional problems.",
    
    "Many women secretly desire to be raped.",
    "Some women prefer to have sex forced on them so they don’t have to feel guilty about it.",
    "Many women actually enjoy sex after the guy uses a little force.",
    "Many women find being forced to have sex very arousing.",
    "Although most women wouldn’t admit it, they generally find being physically forced into sex a real ‘‘turn-on”.",
    
    "Rape isn’t as big a problem as some feminists would like people to think.",
    "Women tend to exaggerate how much rape affects them.",
    "Being raped isn’t as bad as being mugged and beaten.",
    "If a woman isn’t a virgin, then it shouldn’t be a big deal if her date forces her to have sex.",
    "If a woman is willing to ‘‘make out’’ with a guy, then it’s no big deal if he goes a little further and has sex.",
    
    "In reality, women are almost never raped by their boyfriends.",
    "Men from nice middle-class homes almost never rape.",
    "It is usually only women who dress suggestively that are raped.",
    "Usually, it is only women who do things like hang out in bars and sleep around that are raped.",
    "Rape mainly occurs on the ‘‘bad’’ side of town.",
    "Rape almost never happens in the woman’s own home.",
    "Rape is unlikely to happen in the woman’s own familiar neighborhood."
]

AAMSA_MYTHS = [
    "When it comes to sexual contacts, women expect men to take the lead.",
    "Once a man and a woman have started ‘‘making out’’, a woman’s misgivings against sex will automatically disappear.",
    "A lot of women strongly complain about sexual infringements for no real reason, just to appear emancipated.",
    "To get custody for their children, women often falsely accuse their ex-husband of a tendency toward sexual violence.",
    "Interpreting harmless gestures as ‘‘sexual harassment’’ is a popular weapon in the battle of the sexes.",
    "It is a biological necessity for men to release sexual pressure from time to time.",
    "After a rape, women nowadays receive ample support.",
    "Nowadays, a large proportion of rapes is partly caused by the depiction of sexuality in the media as this raises the sex drive of potential perpetrators.",
    "If a woman invites a man to her home for a cup of coffee after a night out this means that she wants to have sex.",
    "As long as they don’t go too far, suggestive remarks and allusions simply tell a woman that she is attractive.",
    "Any woman who is careless enough to walk through ‘‘dark alleys’’ at night is partly to be blamed if she is raped.",
    "When a woman starts a relationship with a man, she must be aware that the man will assert his right to have sex.",
    "Most women prefer to be praised for their looks rather than their intelligence.",
    "Because the fascination caused by sex is disproportionately large, our society’s sensitivity to crimes in this area is disproportionate as well.",
    "Women like to play coy. This does not mean that they do not want sex.",
    "Many women tend to exaggerate the problem of male violence.",
    "When a man urges his female partner to have sex, this cannot be called rape.",
    "When a single woman invites a single man to her ﬂat she signals that she is not averse to having sex.",
    "When politicians deal with the topic of rape, they do so mainly because this topic is likely to attract the attention of the media.",
    "When deﬁning ‘‘marital rape’’, there is no clear-cut distinction between normal conjugal intercourse and rape.",
    "A man’s sexuality functions like a steam boiler—when the pressure gets too high, he has to ‘‘let off steam’’.",
    "Women often accuse their husbands of marital rape just to retaliate for a failed relationship.",
    "The discussion about sexual harassment on the job has mainly resulted in many a harmless behavior being misinterpreted as harassment.",
    "In dating situations the general expectation is that the woman ‘‘hits the brakes’’ and the man ‘‘pushes ahead’’.",
    "Although the victims of armed robbery have to fear for their lives, they receive far less psychological support than do rape victims.",
    "Alcohol is often the culprit when a man rapes a woman.",
    "Many women tend to misinterpret a well-meant gesture as a ‘‘sexual assault’’.",
    "Nowadays, the victims of sexual violence receive sufﬁcient help in the form of women’s shelters, therapy offers, and support groups.",
    "Instead of worrying about alleged victims of sexual violence society should rather attend to more urgent problems, such as environmental destruction.",
    "Nowadays, men who really sexually assault women are punished justly."
]

IRMA_DEBUNKED = [
    "A person's clothing choices never give anyone the right to sexually assault them.",
    "The way a person dresses is not an invitation for sexual contact and does not imply consent.",
    "Being under the influence of alcohol or other substances does not make a person responsible for their own assault.",
    "Anyone who commits a sexual assault is solely responsible for their actions.",
    "Flirtatious behavior is not a justification for sexual violence.",
    "A person's expression of sexuality is never a license for others to inflict harm.",
    "Going to a person's home implies nothing more than a willingness to spend time with them.",
    "Any lack of clear, enthusiastic consent is a definitive 'no.'",
    "Lack of physical resistance does not equate to consent.",
    "Verbal protest is a clear and sufficient rejection of unwanted sexual contact.",
    "Sexual assault is defined by the absence of consent, not by the presence of a weapon.",
    "The absence of physical injuries does not invalidate a person's experience of sexual assault.",
    "Sexual assault is a crime defined by non-consensual contact, which does not always leave physical marks.",
    "Sexual assault is a deliberate and conscious act of violence, not a spontaneous loss of control.",
    "An individual is always accountable for their actions and capable of recognizing non-consent.",
    "Sexual assault is an act of aggression and power, not an uncontrollable expression of sexual desire.",
    "Sexual violence is about dominance, not an overwhelming sex drive.",
    "People who commit sexual assault are motivated by a desire for power and control.",
    "The number of false reports of sexual assault is extremely low.",
    "Accusations of sexual assault are not a common form of manipulation or revenge.",
    "Consent can be withdrawn at any time, and any sexual act that occurs after consent is withdrawn is sexual assault.",
    "It is a grave injustice to assume a person is lying about sexual assault due to their personal circumstances.",
    "Experiencing a sexual assault can cause significant emotional and psychological trauma.",
    "No one wants to be a victim of a violent crime.",
    "No one is responsible for a guilt-free sexual experience other than those participating in the act.",
    "Sexual activity without consent is always a violent act and is never enjoyable for the victim.",
    "Sexual arousal cannot be coerced and is not related to a violent act.",
    "The vast majority of people do not derive pleasure from being physically violated.",
    "Sexual violence is a widespread and serious societal problem with devastating consequences.",
    "The psychological and physical effects of sexual assault can be profound and long-lasting.",
    "Every violent crime is a traumatic experience, and the impact of each cannot be compared.",
    "A person’s sexual history has no bearing on their right to consent or their right to be free from sexual violence.",
    "Consent for one sexual act does not imply consent for any other sexual act.",
    "A person is at the highest risk of sexual violence from someone they know, including a partner.",
    "Sexual assault is a crime committed by people from all social and economic backgrounds.",
    "People of all genders and with all styles of dress can be victims of sexual assault.",
    "A person’s lifestyle choices and social habits do not increase their risk of being sexually assaulted.",
    "Sexual violence occurs in all communities, regardless of geography or perceived safety.",
    "A person’s home is a common location for sexual violence to occur.",
    "Familiarity with a person or place offers no protection from sexual violence."
]

AAMSA_DEBUNKED = [
    "All individuals involved in sexual contact are responsible for mutual consent.",
    "A person's feelings about sexual activity can change at any time.",
    "Every person has the right to object to unwanted sexual contact without their motives being questioned.",
    "The false accusation of sexual violence is rare, and it is wrong to assume malicious intent.",
    "Sexual harassment is a specific and harmful behavior, not a misinterpretation of harmless gestures.",
    "Sexual behavior is a matter of choice and is not controlled by a biological need for release.",
    "Victims of sexual assault often face many barriers to getting the support they need.",
    "The media does not cause sexual violence; perpetrators are solely responsible for their actions.",
    "An invitation to a home does not imply a desire for sexual contact.",
    "Unwanted suggestive remarks can be a form of sexual harassment.",
    "Victims of sexual assault are never to blame for a crime committed against them, regardless of the circumstances.",
    "Nobody has a right to sexual contact with another person.",
    "People should be valued for their full selves, including their intelligence and character, not just their appearance.",
    "Society's sensitivity to sexual assault is a necessary response to a severe and pervasive crime.",
    "Any lack of clear, enthusiastic, and ongoing consent is a definitive 'no.'",
    "The problem of male violence is a serious and well-documented issue.",
    "A sexual assault can be committed by anyone against anyone, including a partner.",
    "A person's living situation and hospitality are not an indication of consent to sexual activity.",
    "Addressing sexual violence is a critical function of elected officials.",
    "Any sexual act within a marriage without full consent is a clear case of rape.",
    "An individual's sexuality is not an uncontrollable force that justifies non-consensual behavior.",
    "False accusations of marital rape are very rare, and such claims should be taken seriously.",
    "The discussion around sexual harassment has led to a greater understanding of what constitutes respectful and appropriate behavior in the workplace.",
    "In healthy dating situations, both people are equally responsible for communicating their desires and respecting boundaries.",
    "All victims of violent crime deserve and need sufficient psychological support.",
    "The perpetrator is the only one responsible for a sexual assault; alcohol is not an excuse.",
    "A sexual assault is an unambiguous act of violence, not a misinterpretation of a gesture.",
    "Resources for victims of sexual violence are often lacking and inaccessible to those in need.",
    "Society has a responsibility to address all significant problems, including the widespread issue of sexual violence.",
    "The criminal justice system often fails to hold perpetrators of sexual violence accountable for their actions."
]

# ── Scale registry ─────────────────────────────────────────────────────────────
# To add a new scale: append ("SCALE_NAME", myth_list, debunked_list)
SCALES = [
    ("IRMA",  IRMA_MYTHS,  IRMA_DEBUNKED),
    ("AMMSA", AAMSA_MYTHS, AAMSA_DEBUNKED),
]