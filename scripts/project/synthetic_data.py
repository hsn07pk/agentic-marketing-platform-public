import pandas as pd
import random
from faker import Faker
import numpy as np

fake = Faker()

platforms = ['linkedin', 'facebook', 'instagram', 'google', 'tiktok']
personas = ['decision_maker', 'manager', 'engineer', 'marketer', 'HR']
campaign_types = ['Boost Event Engagement', 'Lead Gen', 'Product Launch', 'Webinar Invite', 'Brand Awareness']
content_types = ['ad_copy', 'video', 'image']

data = []

for i in range(1000):  # generate 1000 rows
    campaign_id = f"CAMP_{random.choice(platforms)[:2].upper()}_{fake.date_between(start_date='-3M', end_date='today').strftime('%Y%m%d')}_{i}"
    campaign_name = random.choice(campaign_types)
    platform = random.choice(platforms)
    persona = random.choice(personas)
    date = fake.date_between(start_date='-3M', end_date='today')
    
    impressions = random.randint(500, 50000)
    clicks = random.randint(0, int(impressions * 0.2))
    conversions = random.randint(0, clicks)
    ctr = round(clicks / impressions if impressions else 0, 4)
    budget_spent = round(random.uniform(50, 5000), 2)
    cpl = round(budget_spent / conversions if conversions else budget_spent, 2)
    
    headline = fake.sentence(nb_words=10)
    ad_text = fake.paragraph(nb_sentences=3)
    content_type = random.choice(content_types)
    
    data.append([
        campaign_id, campaign_name, platform, persona, date, impressions, clicks, conversions,
        ctr, cpl, budget_spent, headline, ad_text, content_type
    ])

df = pd.DataFrame(data, columns=['campaign_id','campaign_name','platform','persona','date','impressions','clicks','conversions','ctr','cpl','budget_spent','headline','ad_text','content_type'])

# Save as CSV
df.to_csv('campaign_results.csv', index=False)
print("Synthetic campaign data generated!")
