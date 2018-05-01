import pandas as pd
import json

class DataSource:
    ShopName = {}
    ShopToken = {}
    TagIdName = {}
    TagShopOptionGroup = {}
    ShopCrowdTagInfo = 0
    ShopCrowdSelects = 0
    ShopCrowdReplaceRank = 0
    ShopCrowdReplace = 0
    CrowdRate = 0
    CampaignIdPriceModel = 0

    def LoadTagIdName(self, DataFileName):
        dicList = [json.loads(line) for line in open(DataFileName)]
        self.TagIdName = dicList[0]

    def LoadShopName(self,DataFileName):
        dicList = [json.loads(line) for line in open(DataFileName)]
        self.ShopName = dicList[0]

    def LoadShopToken(self,DataFileName):
        dicList = [json.loads(line) for line in open(DataFileName)]
        self.ShopToken = dicList[0]

    ##Tag-Shop-{values,option_group,tag_id
    def LoadTagShopOptionGroup(self, DataFileName):
        dicList = [json.loads(line) for line in open(DataFileName)]
        self.TagShopOptionGroup = dicList[0]




TagIdNameFile = "./DataLocation/TagIdName0328.json"
ShopNameFile =  "./DataLocation/ShopName0328.json"
ShopTokenFile = "./DataLocation/shopToken0328.json"
TagShopOptionGroupFile = "./DataLocation/TagShopOptionGroup0328.json"
a = DataSource()
a.LoadTagIdName(TagIdNameFile)
#a.LoadShopName(ShopNameFile)
#a.LoadShopToken(ShopTokenFile)
#a.LoadTagShopOptionGroup(TagShopOptionGroupFile)
