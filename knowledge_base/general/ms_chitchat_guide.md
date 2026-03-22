# Microsoft Learn - Chit-chat knowledge base

- Source: https://learn.microsoft.com/azure/ai-services/qnamaker/how-to/chit-chat-knowledge-base
- Downloaded at (UTC): 2026-03-22T08:51:24.273556+00:00

---

#### Share via
Note
Access to this page requires authorization. You can try signing in or changing directories .
Access to this page requires authorization. You can try changing directories .
# Add Chit-chat to a knowledge base
Adding chit-chat to your bot makes it more conversational and engaging. The chit-chat feature in QnA maker allows you to easily add a pre-populated set of the top chit-chat, into your knowledge base (KB). This can be a starting point for your bot's personality, and it will save you the time and cost of writing them from scratch.
Note
The QnA Maker service is being retired on the October 31, 2025 (extended from March 31, 2025). A newer version of the question and answering capability is now available as part of Azure Language . For question answering capabilities within Azure Language Service, see custom question answering (CQA) . As of October 1, 2022, you're no longer able to create new QnA Maker resources. Beginning on March 31, 2025, the QnA Maker portal is no longer available. For information on migrating existing QnA Maker knowledge bases to question answering, consult the migration guide .
This dataset has about 100 scenarios of chit-chat in the voice of multiple personas, like Professional, Friendly and Witty. Choose the persona that most closely resembles your bot's voice. Given a user query, QnA Maker tries to match it with the closest known chit-chat QnA.
Some examples of the different personalities are below. You can see all the personality datasets along with details of the personalities.
For the user query of When is your birthday? , each personality has a styled response:
## Language support
Chit-chat data sets are supported in the following languages:
## Add chit-chat during KB creation
During knowledge base creation, after adding your source URLs and files, there is an option for adding chit-chat. Choose the personality that you want as your chit-chat base. If you do not want to add chit-chat, or if you already have chit-chat support in your data sources, choose None .
## Add Chit-chat to an existing KB
Select your KB, and navigate to the Settings page. There is a link to all the chit-chat datasets in the appropriate .tsv format. Download the personality you want, then upload it as a file source. Make sure not to edit the format or the metadata when you download and upload the file.
## Edit your chit-chat questions and answers
When you edit your KB, you will see a new source for chit-chat, based on the personality you selected. You can now add altered questions or edit the responses, just like with any other source.
To view the metadata, select View Options in the toolbar, then select Show metadata .
## Add additional chit-chat questions and answers
You can add a new chit-chat QnA pair that is not in the predefined data set. Ensure that you are not duplicating a QnA pair that is already covered in the chit-chat set. When you add any new chit-chat QnA, it gets added to your Editorial source. To ensure the ranker understands that this is chit-chat, add the metadata key/value pair "Editorial: chitchat", as seen in the following image:
## Delete chit-chat from an existing KB
Select your KB, and navigate to the Settings page. Your specific chit-chat source is listed as a file, with the selected personality name. You can delete this as a source file.
## Next steps
Import a knowledge base
## See also
QnA Maker overview
## Additional resources
- Last updated on 2025-06-12


---

## Local fallback templates for daily conversation

- Greeting: 你好呀，我在的。今天想聊训练、饮食，还是先随便聊聊？
- Thanks: 不客气～如果你愿意，我可以继续帮你把今天的计划细化成 3 个可执行步骤。
- Goodbye: 好的，随时来找我。祝你今天顺利，晚点我们再继续。
- Light support: 辛苦了，我们先做一个最小动作，比如 10 分钟散步或喝杯水。
