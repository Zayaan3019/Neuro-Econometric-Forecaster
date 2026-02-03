"""
Data Loader Module for Market Alpha Engine.

This module handles fetching OHLCV data from Yahoo Finance and financial news
from NewsAPI, providing a unified interface for multi-modal data retrieval.
"""

from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import time
import pandas as pd
import numpy as np
import yfinance as yf
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

from app.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OHLCVLoader:
    """
    Fetches OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance.
    
    Implements robust retry logic and data validation to ensure high-quality
    market data for downstream modeling.
    """
    
    def __init__(self, ticker: str = Config.TICKER, local_csv: Optional[str] = None):
        """
        Initialize OHLCV loader.
        
        Args:
            ticker: Stock/Index ticker symbol (e.g., '^GSPC', 'BTC-USD').
            local_csv: Optional path to local CSV file as fallback data source.
        """
        self.ticker = ticker
        self.local_csv = local_csv
        self._setup_session()
    
    def _setup_session(self) -> None:
        """Configure HTTP session with aggressive retry strategy for robust API calls."""
        self.session = Session()
        retry_strategy = Retry(
            total=5,  # Increased retries
            backoff_factor=2,  # More aggressive backoff (2, 4, 8, 16, 32 seconds)
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def fetch(
        self,
        start_date: str = Config.START_DATE,
        end_date: str = Config.END_DATE,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for the specified date range.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            interval: Data frequency ('1d' for daily, '1h' for hourly).
        
        Returns:
            DataFrame with columns: [Open, High, Low, Close, Volume, Adj Close].
            Index is DatetimeIndex.
        
        Raises:
            ValueError: If data fetching fails or returns empty dataset.
        """
        # Try local CSV first if provided
        if self.local_csv and Path(self.local_csv).exists():
            try:
                logger.info(f"Loading OHLCV data from local CSV: {self.local_csv}")
                return self._load_from_csv(start_date, end_date)
            except Exception as e:
                logger.warning(f"Failed to load from local CSV: {str(e)}. Falling back to API.")
        
        logger.info(f"Fetching OHLCV data for {self.ticker} from {start_date} to {end_date}")
        
        # Multiple retry attempts with exponential backoff
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    wait_time = 15 * (2 ** attempt)  # 30s, 60s
                    logger.info(f"Retry attempt {attempt + 1}/{max_attempts} after {wait_time}s wait...")
                    time.sleep(wait_time)
                
                ticker_obj = yf.Ticker(self.ticker, session=self.session)
                df = ticker_obj.history(start=start_date, end=end_date, interval=interval)
                
                if df.empty:
                    if attempt < max_attempts - 1:
                        continue
                    raise ValueError(f"No data returned for {self.ticker}")
                
                # Data validation
                df = self._validate_and_clean(df)
                
                logger.info(f"Successfully fetched {len(df)} rows of OHLCV data")
                return df
            
            except Exception as e:
                if attempt == max_attempts - 1:
                    logger.error(f"All {max_attempts} attempts failed. Error: {str(e)}")
                    raise
                else:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    continue
    
    def _load_from_csv(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load OHLCV data from local CSV file.
        
        Args:
            start_date: Filter start date in 'YYYY-MM-DD' format.
            end_date: Filter end date in 'YYYY-MM-DD' format.
        
        Returns:
            DataFrame with OHLCV data matching yfinance format.
        """
        df = pd.read_csv(self.local_csv)
        
        # Try to identify date column
        date_col = None
        for col in ['Date', 'date', 'Datetime', 'datetime', 'timestamp', 'TIME']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            # Assume first column is date
            date_col = df.columns[0]
            logger.warning(f"No standard date column found, using first column: {date_col}")
        
        # Parse dates
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df.index.name = 'Date'
        
        # Standardize column names (case-insensitive)
        df.columns = [col.title() for col in df.columns]
        
        # Ensure required columns exist
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        
        # Filter by date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        df = df[(df.index >= start) & (df.index <= end)]
        
        if df.empty:
            raise ValueError(f"No data in CSV for date range {start_date} to {end_date}")
        
        logger.info(f"Loaded {len(df)} rows from local CSV: {self.local_csv}")
        return self._validate_and_clean(df)
    
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean OHLCV data.
        
        Args:
            df: Raw OHLCV DataFrame.
        
        Returns:
            Cleaned DataFrame with missing values handled.
        """
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Forward-fill missing values (conservative approach for financial data)
        df = df.fillna(method='ffill')
        
        # Ensure no negative prices or volumes
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        
        # Ensure High >= Low
        if 'High' in df.columns and 'Low' in df.columns:
            df['High'] = df[['High', 'Low']].max(axis=1)
        
        return df


class NewsLoader:
    """
    Fetches financial news headlines using NewsAPI.
    
    Provides time-aligned news data that can be merged with OHLCV data for
    sentiment analysis in the hybrid model.
    """
    
    def __init__(self, api_key: str = Config.NEWS_API_KEY):
        """
        Initialize News loader.
        
        Args:
            api_key: NewsAPI authentication key.
        """
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
        self._setup_session()
    
    def _setup_session(self) -> None:
        """Configure HTTP session with retry strategy."""
        self.session = Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def fetch(
        self,
        query: str = "stock market OR S&P 500 OR economy",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        language: str = "en",
        sort_by: str = "relevancy"
    ) -> pd.DataFrame:
        """
        Fetch financial news headlines for the specified date range.
        
        Args:
            query: Search query for news articles.
            start_date: Start date in 'YYYY-MM-DD' format (defaults to 30 days ago).
            end_date: End date in 'YYYY-MM-DD' format (defaults to today).
            language: Language code (e.g., 'en' for English).
            sort_by: Sorting criterion ('relevancy', 'popularity', 'publishedAt').
        
        Returns:
            DataFrame with columns: [date, title, description, source, url].
            Index is DatetimeIndex aligned with trading days.
        
        Raises:
            ValueError: If API key is invalid or request fails.
        """
        if self.api_key == "YOUR_NEWS_API_KEY_HERE":
            logger.warning("NewsAPI key not configured. Returning empty DataFrame.")
            return self._create_empty_news_df()
        
        # Default date range: last 30 days
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        logger.info(f"Fetching news from {start_date} to {end_date}")
        
        all_articles = []
        page = 1
        
        try:
            while True:
                params = {
                    "q": query,
                    "from": start_date,
                    "to": end_date,
                    "language": language,
                    "sortBy": sort_by,
                    "apiKey": self.api_key,
                    "page": page,
                    "pageSize": 100
                }
                
                response = self.session.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") != "ok":
                    raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                
                articles = data.get("articles", [])
                if not articles:
                    break
                
                all_articles.extend(articles)
                
                # Check if there are more pages
                if len(all_articles) >= data.get("totalResults", 0):
                    break
                
                page += 1
            
            if not all_articles:
                logger.warning("No news articles found.")
                return self._create_empty_news_df()
            
            # Parse into DataFrame
            df = pd.DataFrame([
                {
                    "date": pd.to_datetime(article["publishedAt"]),
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "url": article.get("url", "")
                }
                for article in all_articles
            ])
            
            df = df.sort_values("date").reset_index(drop=True)
            logger.info(f"Successfully fetched {len(df)} news articles")
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching news data: {str(e)}")
            return self._create_empty_news_df()
    
    def _create_empty_news_df(self) -> pd.DataFrame:
        """Create an empty news DataFrame with correct schema."""
        return pd.DataFrame(columns=["date", "title", "description", "source", "url"])
    
    def aggregate_by_day(
        self,
        news_df: pd.DataFrame,
        max_per_day: int = Config.MAX_NEWS_PER_DAY
    ) -> pd.DataFrame:
        """
        Aggregate news articles by trading day.
        
        Args:
            news_df: Raw news DataFrame with datetime index.
            max_per_day: Maximum number of headlines to keep per day.
        
        Returns:
            DataFrame with one row per day, containing aggregated headlines.
        """
        if news_df.empty:
            return news_df
        
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.date
        
        aggregated = (
            news_df.groupby("date")
            .apply(lambda x: " | ".join(x["title"].head(max_per_day)))
            .reset_index()
        )
        aggregated.columns = ["date", "headlines"]
        aggregated["date"] = pd.to_datetime(aggregated["date"])
        
        return aggregated


class DataLoader:
    """
    Unified data loader combining OHLCV and News data.
    
    Provides a single interface for fetching and merging multi-modal data
    required by the Neuro-Econometric Engine.
    """
    
    def __init__(
        self,
        ticker: str = Config.TICKER,
        news_api_key: str = Config.NEWS_API_KEY,
        local_csv: Optional[str] = None
    ):
        """
        Initialize unified data loader.
        
        Args:
            ticker: Stock/Index ticker symbol.
            news_api_key: NewsAPI authentication key.
            local_csv: Optional path to local CSV file for OHLCV data.
        """
        self.ohlcv_loader = OHLCVLoader(ticker, local_csv=local_csv)
        self.news_loader = NewsLoader(news_api_key)
    
    def load_all(
        self,
        start_date: str = Config.START_DATE,
        end_date: str = Config.END_DATE,
        include_news: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both OHLCV and News data.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            include_news: Whether to fetch news data.
        
        Returns:
            Tuple of (ohlcv_df, news_df).
        """
        # Fetch OHLCV data
        ohlcv_df = self.ohlcv_loader.fetch(start_date, end_date)
        
        # Fetch news data
        if include_news:
            news_df = self.news_loader.fetch(start_date=start_date, end_date=end_date)
            news_df = self.news_loader.aggregate_by_day(news_df)
        else:
            news_df = pd.DataFrame()
        
        return ohlcv_df, news_df
