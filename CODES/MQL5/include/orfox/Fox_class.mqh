//+------------------------------------------------------------------+
//|                                                    Fox_class.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

#define LinkToIndicators1 "OrFox\\Inditest01.ex5"
#define LinkToIndicators2 "OrFox\\212-03.ex5"
#define LinkToIndicators3 "SwimUp7days\\swimUp7Days.ex5"
#define LinkToIndicators4 "OrFox\\closecheck.ex5"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class Fox_class
  {
private:
   int               Magic_number;
   int               Check_margin;
   double            Lots;
   double            TradePct;

   double            Indi_01;
   int               Indi_01_handle;
   double            Indi_01_val[];

   double            Indi_02;
   int               Indi_02_handle;
   double            Indi_02_val[];
   
   double            Indi_03;
   int               Indi_03_handle;
   double            Indi_03_val[];   
   
   double            Indi_04;
   int               Indi_04_handle;
   double            Indi_04_val[];   

   double            ADX_min;
   int               ADX_handle;
   double            ADX_val[];
   double            plus_DI[];
   double            minus_DI[];

   int               MA_handle;
   double            MA_val[];
   
           

   double            ClosePrice;
   MqlTradeRequest   tradeQuest;
   MqlTradeResult    tradeResul;
   string            symbol;
   ENUM_TIMEFRAMES   period;
   string            ErMsg;
   int               ErCode;

public:
   void              Fox_class();
   void             ~Fox_class();
   void              setSymbol(string syb) {symbol = syb;}
   void              setPeriod(ENUM_TIMEFRAMES per) {per = period;}
   void              setClose(double closeprice) {closeprice = ClosePrice;}
   void              setCheckMargin(int mag) {Check_margin=mag;}
   void              setLots(double lots) {Lots=lots;}
   void              setTRpct(double tradePercentages) {TradePct=tradePercentages/100;}
   void              setMagic(int magic) {Magic_number=magic;}
   void              setIndi01(double ind_01) {Indi_01=ind_01;}
   void              setIndi03(double ind_03) {Indi_03=ind_03;}
   void              setIndi04(double ind_04) {Indi_04=ind_04;}
   void              doInit(int Indicator_01, int adx_period, int ma_period, int Indicator_03, int Indicator_04);
   void              checkIndicator(int Indicator);
   void              doUnInit();
   bool              CheckBuy();
   bool              CheckSell();
   void              InfoBuy(MqlTradeRequest &info,MqlTradeResult &info2);
   void              InfoSell(MqlTradeRequest &info,MqlTradeResult &info2);
   void              openBuy(ENUM_ORDER_TYPE odertype,double askprice, double SL, double TP,int dev,string comment="");
   void              openSell(ENUM_ORDER_TYPE odertype,double bidprice, double SL, double TP,int dev,string comment="");
   void              checkHandle(int handle);

protected:
   void              showErrors(string messages, int errcode);
   void              getBuffers();
   bool              MarginOK();
  };

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Fox_class::Fox_class()
  {
   ZeroMemory(tradeQuest);
   ZeroMemory(tradeResul);
   ZeroMemory(Indi_01_val);
   ZeroMemory(Indi_02_val);
   ZeroMemory(Indi_03_val);
   ZeroMemory(Indi_04_val); 
   ZeroMemory(ADX_val);
   ZeroMemory(plus_DI);
   ZeroMemory(minus_DI);
   ZeroMemory(MA_val);
   ErMsg="";
   ErCode=0;
  }

void Fox_class::~Fox_class() {}

void Fox_class::showErrors(string messages,int errcode) {Alert(messages,"\n\t--- Mã lỗi:",errcode);}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Fox_class::getBuffers(void)
  {
   if(CopyBuffer(Indi_01_handle,0,0,10,Indi_01_val)<=0)
     {
      ErMsg = "Copy values of indicator trung binh open/close is not enough";
      ErCode=GetLastError();
      showErrors(ErMsg,ErCode);
     }

   if(CopyBuffer(Indi_02_handle,0,0,10,Indi_02_val)<=0)
     {
      ErMsg = "Copy values of indicator 212 is not enough";
      ErCode=GetLastError();
      showErrors(ErMsg,ErCode);
     }
     
   if(CopyBuffer(Indi_03_handle,0,0,10,Indi_03_val)<=0)
     {
      ErMsg = "Copy values of indicator 7 days is not enough";
      ErCode=GetLastError();
      showErrors(ErMsg,ErCode);
     }  
     
   if(CopyBuffer(Indi_04_handle,0,0,2,Indi_04_val)<=0)
     {
      ErMsg = "Copy values of indicator 7 days is not enough";
      ErCode=GetLastError();
      showErrors(ErMsg,ErCode);
     }              
        
   if(CopyBuffer(ADX_handle,0,0,3,ADX_val)<=0 || CopyBuffer(ADX_handle,1,0,3,plus_DI)<=0 || CopyBuffer(ADX_handle,2,0,3,minus_DI)<=0)
     {
      ErMsg="Error copying indicator ADX Buffers";
      ErCode = GetLastError();
      showErrors(ErMsg,ErCode);
     }

   if(CopyBuffer(MA_handle,0,0,3,MA_val)<=0)
     {
      ErMsg="Error copying indicator MA Buffers";
      ErCode = GetLastError();
      showErrors(ErMsg,ErCode);
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool Fox_class::MarginOK()
  {
   double one_lot_price;
   double act_f_mag  =  AccountInfoDouble(ACCOUNT_FREEMARGIN);
   long   levrage    =  AccountInfoInteger(ACCOUNT_LEVERAGE);
   double contract_size = SymbolInfoDouble(symbol,SYMBOL_TRADE_CONTRACT_SIZE);
   string base_currency = SymbolInfoString(symbol,SYMBOL_CURRENCY_BASE);

   if(base_currency == "USD")
     {
      one_lot_price = contract_size/levrage;
     }
   else
     {
      double bprice = SymbolInfoDouble(symbol,SYMBOL_BID);
      one_lot_price = bprice*contract_size/levrage;
     }

   if(MathFloor(Lots*one_lot_price)>MathFloor(act_f_mag*TradePct))
     {
      return(false);
     }
   else
     {
      return(true);
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Fox_class::checkIndicator(int Indicator)
  {
   if(Indicator<0)
     {
      ErMsg="Fail to creating Indicator";
      ErCode=GetLastError();
      showErrors(ErMsg,ErCode);
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Fox_class::checkHandle(int handle)
  {
   if(handle < 0)
     {
      ErMsg="Error creating handle "+handle;
      ErCode=GetLastError();
      showErrors(ErMsg,ErCode);
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Fox_class::doInit(int Indicator_01, int adx_period, int ma_period, int Indicator_03, int Indicator_04)// thêm indicator ở đây
  {
   Indi_01_handle = iCustom(symbol,period,LinkToIndicators1);
   this.checkIndicator(Indi_01_handle);
   checkHandle(Indi_01_handle);
   ArraySetAsSeries(Indi_01_val,true);

   Indi_02_handle = iCustom(symbol,period,LinkToIndicators2);
   this.checkIndicator(Indi_02_handle);
   checkHandle(Indi_02_handle);
   ArraySetAsSeries(Indi_02_val,true);

   ADX_handle = iADX(symbol,period,adx_period);
   checkHandle(ADX_handle);
   ArraySetAsSeries(ADX_val,true);

   MA_handle  = iMA(symbol,period,ma_period,0,MODE_EMA,PRICE_CLOSE);
   checkHandle(MA_handle);
   ArraySetAsSeries(MA_val,true);
   
   Indi_03_handle = iCustom(symbol,period,LinkToIndicators3);
   this.checkIndicator(Indi_03_handle);
   checkHandle(Indi_03_handle);
   ArraySetAsSeries(Indi_03_val,true);
   
   Indi_04_handle = iCustom(symbol,period,LinkToIndicators4);
   this.checkIndicator(Indi_04_handle);
   checkHandle(Indi_04_handle);
   ArraySetAsSeries(Indi_04_val,true);    
  }

void Fox_class::doUnInit() {IndicatorRelease(Indi_01_handle); IndicatorRelease(Indi_02_handle); IndicatorRelease(ADX_handle); IndicatorRelease(MA_handle); IndicatorRelease(Indi_03_handle); IndicatorRelease(Indi_04_handle);}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool Fox_class::CheckBuy()
  {
   getBuffers();
   bool Buy_01 = (MA_val[0]>MA_val[1]) && (MA_val[1]>MA_val[2]);
   bool Buy_02 = (ClosePrice>MA_val[1]);
   bool Buy_03 = (ADX_val[0]>ADX_min);
   bool Buy_04 = (plus_DI[0]>minus_DI[0]);
   bool Buy_05 = (Indi_01_val[7]>Indi_01_val[6] && Indi_01_val[6]<Indi_01_val[5] && Indi_01_val[5]<Indi_01_val[4] && Indi_01_val[4]>Indi_01_val[3] && Indi_01_val[3]<Indi_01_val[2] && Indi_01_val[2]<Indi_01_val[1]);
   bool Buy_06 = (Indi_01_val[1]<Indi_01_val[2] && Indi_01_val[2]<Indi_01_val[3] && Indi_01_val[3]<Indi_01_val[4] && Indi_01_val[4]<Indi_01_val[5]);

   if(Buy_01 && Buy_02 && Buy_03 && Buy_04)
     {
      if(Buy_05 || Buy_06)
        {
         return true;
        }
      return false;
     }
   else
     {
      return false;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool Fox_class::CheckSell()
  {
   getBuffers();
   bool Sell_01 = (MA_val[0]<MA_val[1]) && (MA_val[1]<MA_val[2]);
   bool Sell_02 = (ClosePrice <MA_val[1]);
   bool Sell_03 = (ADX_val[0]>ADX_min);
   bool Sell_04 = (plus_DI[0]<minus_DI[0]);
   bool Sell_05 = (Indi_01_val[7]<Indi_01_val[6] && Indi_01_val[6]>Indi_01_val[5] && Indi_01_val[5]>Indi_01_val[4] && Indi_01_val[4]<Indi_01_val[3] && Indi_01_val[3]>Indi_01_val[2] && Indi_01_val[2]>Indi_01_val[1]);
   bool Sell_06 = (Indi_01_val[1]>Indi_01_val[2] && Indi_01_val[2]>Indi_01_val[3] && Indi_01_val[3]>Indi_01_val[4] && Indi_01_val[4]>Indi_01_val[5]);
   if(Sell_01 && Sell_02 && Sell_03 && Sell_04)
     {
      if(Sell_05 || Sell_06)
        {
         return true;
        }
      return false;
     }
   else
     {
      return false;
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Fox_class::InfoBuy(MqlTradeRequest &info,MqlTradeResult &info2)
  {
   if(!OrderSend(info,info2))
     {
      Alert("The Buy order request could not be completed: ",GetLastError());
     }
   else
     {
      Alert("--->> Giao dich Buy thanh cong !!! ","Deal: ",info2.deal," Order: ",info2.order," RetCode: ",info2.retcode);
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Fox_class::InfoSell(MqlTradeRequest &info,MqlTradeResult &info2)
  {
   if(!OrderSend(info,info2))
     {
      Alert("The Sell order request could not be completed: ",GetLastError());
     }
   else
     {
      Alert("--->> Giao dich Sell thanh cong !!! ","Deal: ",info2.deal," Order: ",info2.order," RetCode: ",info2.retcode);
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Fox_class::openBuy(ENUM_ORDER_TYPE odertype,double askprice,double SL,double TP,int dev,string comment="")
  {
   if(Check_margin==1)
     {
      if(MarginOK()==false)
        {
         ErMsg="We don't have enough money to open this Position";
         ErCode=GetLastError();
         showErrors(ErMsg,ErCode);
        }
      else
        {
         tradeQuest.action =  TRADE_ACTION_DEAL;
         tradeQuest.type   =  odertype;
         tradeQuest.volume =  Lots;
         tradeQuest.price  =  askprice;
         tradeQuest.sl     =  SL;
         tradeQuest.tp     =  TP;
         tradeQuest.deviation=   dev;
         tradeQuest.magic  =  Magic_number;
         tradeQuest.symbol =  symbol;
         tradeQuest.type_filling =  ORDER_FILLING_FOK;
         OrderSend(tradeQuest,tradeResul);
         this.InfoBuy(tradeQuest,tradeResul);
        }
     }
   else
     {
      tradeQuest.action =  TRADE_ACTION_DEAL;
      tradeQuest.type   =  odertype;
      tradeQuest.volume =  Lots;
      tradeQuest.price  =  askprice;
      tradeQuest.sl     =  SL;
      tradeQuest.tp     =  TP;
      tradeQuest.deviation=   dev;
      tradeQuest.magic  =  Magic_number;
      tradeQuest.symbol =  symbol;
      tradeQuest.type_filling =  ORDER_FILLING_FOK;
      OrderSend(tradeQuest,tradeResul);
      this.InfoBuy(tradeQuest,tradeResul);
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Fox_class::openSell(ENUM_ORDER_TYPE odertype,double bidprice,double SL,double TP,int dev,string comment="")
  {
   if(Check_margin==1)
     {
      if(MarginOK()==false)
        {
         ErMsg="We don't have enough money to open this Position";
         ErCode=GetLastError();
         showErrors(ErMsg,ErCode);
        }
      else
        {
         tradeQuest.action =  TRADE_ACTION_DEAL;
         tradeQuest.type   =  odertype;
         tradeQuest.volume =  Lots;
         tradeQuest.price  =  bidprice;
         tradeQuest.sl     =  SL;
         tradeQuest.tp     =  TP;
         tradeQuest.deviation=   dev;
         tradeQuest.magic  =  Magic_number;
         tradeQuest.symbol =  symbol;
         tradeQuest.type_filling =  ORDER_FILLING_FOK;
         OrderSend(tradeQuest,tradeResul);
         this.InfoBuy(tradeQuest,tradeResul);
        }
     }
   else
     {
      tradeQuest.action =  TRADE_ACTION_DEAL;
      tradeQuest.type   =  odertype;
      tradeQuest.volume =  Lots;
      tradeQuest.price  =  bidprice;
      tradeQuest.sl     =  SL;
      tradeQuest.tp     =  TP;
      tradeQuest.deviation=   dev;
      tradeQuest.magic  =  Magic_number;
      tradeQuest.symbol =  symbol;
      tradeQuest.type_filling =  ORDER_FILLING_FOK;
      OrderSend(tradeQuest,tradeResul);
      this.InfoBuy(tradeQuest,tradeResul);
     }
  }

//+------------------------------------------------------------------+
