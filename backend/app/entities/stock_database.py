"""
Stock ticker database management.

Provides comprehensive ticker-name mapping for all publicly traded stocks.
Data sources: SEC EDGAR Company Tickers + yfinance enrichment.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlopen

logger = logging.getLogger(__name__)


class StockDatabase:
    """Manages stock ticker database with company name mappings."""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize stock database.

        Args:
            data_dir: Directory to store stock data. Defaults to data/stocks/
        """
        if data_dir is None:
            # Default to data/stocks/ in project root
            backend_dir = Path(__file__).parent.parent.parent
            data_dir = backend_dir.parent / "data" / "stocks"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.database_file = self.data_dir / "stock_database.json"
        self._stocks: Dict[str, Dict] = {}
        self._name_to_ticker: Dict[str, str] = {}
        self._loaded = False

    def load(self) -> None:
        """Load stock database from file or download if not exists."""
        if self._loaded:
            return

        if self.database_file.exists():
            logger.info(f"Loading stock database from {self.database_file}")
            with open(self.database_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._stocks = data.get("stocks", {})
                self._build_name_index()
                self._loaded = True
                logger.info(f"Loaded {len(self._stocks)} stocks from database")
        else:
            logger.info("Stock database not found. Downloading from SEC EDGAR...")
            self.download_and_build()

    def download_and_build(self) -> None:
        """Download stock data from SEC EDGAR and build database."""
        logger.info("Downloading SEC EDGAR company tickers...")

        try:
            # Download SEC EDGAR company tickers JSON
            # SEC requires specific User-Agent with contact info
            url = "https://www.sec.gov/files/company_tickers.json"

            # Try using requests library first (better handling)
            try:
                import requests

                headers = {
                    "User-Agent": "InvestorSentimentDashboard/1.0 (Academic Research; contact@example.com)",
                    "Accept-Encoding": "gzip, deflate",
                    "Accept": "application/json",
                }
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                sec_data = response.json()
            except ImportError:
                # Fall back to urllib if requests not available
                from urllib.request import Request

                req = Request(
                    url,
                    headers={
                        "User-Agent": "InvestorSentimentDashboard/1.0 (Academic Research; contact@example.com)"
                    },
                )
                with urlopen(req) as response:
                    sec_data = json.loads(response.read().decode("utf-8"))

            logger.info(f"Downloaded {len(sec_data)} companies from SEC")

            # Build stock database
            self._stocks = {}
            for item in sec_data.values():
                ticker = item["ticker"]
                company_name = item["title"]

                # Extract common short name (remove Inc., Corp., etc.)
                short_name = self._extract_short_name(company_name)

                self._stocks[ticker] = {
                    "ticker": ticker,
                    "company_name": company_name,
                    "common_names": [short_name, ticker],
                    "exchange": "US",  # SEC data is US companies
                    "cik": item["cik_str"],
                    "is_active": True,
                }

            self._build_name_index()
            self.save()
            self._loaded = True

            logger.info(f"Built stock database with {len(self._stocks)} stocks")

        except Exception as e:
            logger.error(f"Failed to download stock data: {e}")
            logger.info("Trying alternative: building database from common stocks...")

            # Fallback: Create database with common major stocks
            self._build_fallback_database()
            self._build_name_index()
            self.save()
            self._loaded = True

    def save(self) -> None:
        """Save stock database to file."""
        data = {
            "stocks": self._stocks,
            "metadata": {
                "total_stocks": len(self._stocks),
                "source": "SEC EDGAR",
            },
        }

        with open(self.database_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved stock database to {self.database_file}")

    def _extract_short_name(self, company_name: str) -> str:
        """
        Extract short company name by removing common suffixes.

        Args:
            company_name: Full company name

        Returns:
            Short company name
        """
        # Remove common suffixes (including variations with commas, slashes)
        suffixes = [
            # With comma (SEC format)
            ", Inc.",
            ", Inc",
            ", Corp.",
            ", Corp",
            ", Corporation",
            ", Ltd.",
            ", Ltd",
            ", Limited",
            ", LLC",
            ", L.L.C.",
            ", PLC",
            ", Co.",
            # Without comma
            " Inc.",
            " Inc",
            " Corp.",
            " Corp",
            " Corporation",
            " Ltd.",
            " Ltd",
            " Limited",
            " LLC",
            " L.L.C.",
            " PLC",
            " Co.",
            " Company",
            " & Co.",
            # With slash
            "/Inc",
            "/DE",
            "/MD",
            "/NV",
            "/PA",
        ]

        short_name = company_name
        for suffix in suffixes:
            if short_name.lower().endswith(suffix.lower()):
                short_name = short_name[: -len(suffix)]
                break

        # Also remove trailing state codes like " - DE", " - MD"
        import re

        short_name = re.sub(r"\s*[-/]\s*[A-Z]{2}$", "", short_name)

        # Remove "THE " prefix for indexing
        if short_name.upper().startswith("THE "):
            short_name = short_name[4:]

        return short_name.strip()

    def _build_name_index(self) -> None:
        """Build reverse index from company names to tickers."""
        self._name_to_ticker = {}

        for ticker, stock in self._stocks.items():
            # Index by full company name (lowercase)
            company_name = stock["company_name"]
            self._name_to_ticker[company_name.lower()] = ticker

            # Index by common names
            for common_name in stock.get("common_names", []):
                name_lower = common_name.lower()
                if name_lower and name_lower not in self._name_to_ticker:
                    self._name_to_ticker[name_lower] = ticker

            # Extract and index short name
            short_name = self._extract_short_name(company_name)
            if short_name:
                short_lower = short_name.lower()
                if short_lower not in self._name_to_ticker:
                    self._name_to_ticker[short_lower] = ticker

                # Also index title case version for matching
                # e.g., "MICROSOFT" -> "Microsoft"
                title_case = short_name.title()
                if title_case.lower() not in self._name_to_ticker:
                    self._name_to_ticker[title_case.lower()] = ticker

            # Index first word if it's substantial (>3 chars)
            # Helps match "Tesla" from "Tesla, Inc."
            first_word = company_name.split()[0] if company_name else ""
            if len(first_word) > 3 and first_word.lower() not in self._name_to_ticker:
                # Avoid common words
                skip_words = {
                    "the",
                    "inc",
                    "corp",
                    "ltd",
                    "llc",
                    "plc",
                    "new",
                    "first",
                    "american",
                    "national",
                    "united",
                }
                if first_word.lower() not in skip_words:
                    self._name_to_ticker[first_word.lower()] = ticker

    def get_by_ticker(self, ticker: str) -> Optional[Dict]:
        """
        Get stock info by ticker symbol.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            Stock info dict or None if not found
        """
        if not self._loaded:
            self.load()
        return self._stocks.get(ticker.upper())

    def get_by_name(self, name: str) -> Optional[Dict]:
        """
        Get stock info by company name.

        Args:
            name: Company name

        Returns:
            Stock info dict or None if not found
        """
        if not self._loaded:
            self.load()

        ticker = self._name_to_ticker.get(name.lower())
        if ticker:
            return self._stocks.get(ticker)
        return None

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for stocks by name or ticker.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching stock info dicts
        """
        if not self._loaded:
            self.load()

        query_lower = query.lower()
        results = []

        # Exact ticker match
        if query.upper() in self._stocks:
            results.append(self._stocks[query.upper()])

        # Partial name matches
        for name, ticker in self._name_to_ticker.items():
            if query_lower in name and ticker not in [r["ticker"] for r in results]:
                results.append(self._stocks[ticker])

            if len(results) >= limit:
                break

        return results[:limit]

    def get_all_tickers(self) -> List[str]:
        """
        Get list of all ticker symbols.

        Returns:
            List of ticker symbols
        """
        if not self._loaded:
            self.load()
        return list(self._stocks.keys())

    def get_total_stocks(self) -> int:
        """
        Get total number of stocks in database.

        Returns:
            Number of stocks
        """
        if not self._loaded:
            self.load()
        return len(self._stocks)

    def _build_fallback_database(self) -> None:
        """
        Build fallback database with common major stocks.

        Used when SEC EDGAR download fails.
        """
        # Top 100+ most commonly mentioned stocks
        fallback_stocks = [
            # Tech Giants
            ("AAPL", "Apple Inc.", "Apple"),
            ("MSFT", "Microsoft Corporation", "Microsoft"),
            ("GOOGL", "Alphabet Inc.", "Google"),
            ("GOOG", "Alphabet Inc.", "Google"),
            ("AMZN", "Amazon.com, Inc.", "Amazon"),
            ("META", "Meta Platforms, Inc.", "Facebook"),
            ("TSLA", "Tesla, Inc.", "Tesla"),
            ("NVDA", "NVIDIA Corporation", "NVIDIA"),
            ("AMD", "Advanced Micro Devices, Inc.", "AMD"),
            ("INTC", "Intel Corporation", "Intel"),
            # Financial
            ("JPM", "JPMorgan Chase & Co.", "JPMorgan"),
            ("BAC", "Bank of America Corporation", "Bank of America"),
            ("WFC", "Wells Fargo & Company", "Wells Fargo"),
            ("GS", "The Goldman Sachs Group, Inc.", "Goldman Sachs"),
            ("MS", "Morgan Stanley", "Morgan Stanley"),
            ("C", "Citigroup Inc.", "Citigroup"),
            ("V", "Visa Inc.", "Visa"),
            ("MA", "Mastercard Incorporated", "Mastercard"),
            ("AXP", "American Express Company", "American Express"),
            ("BLK", "BlackRock, Inc.", "BlackRock"),
            # Healthcare
            ("JNJ", "Johnson & Johnson", "Johnson & Johnson"),
            ("UNH", "UnitedHealth Group Incorporated", "UnitedHealth"),
            ("PFE", "Pfizer Inc.", "Pfizer"),
            ("ABBV", "AbbVie Inc.", "AbbVie"),
            ("TMO", "Thermo Fisher Scientific Inc.", "Thermo Fisher"),
            ("MRK", "Merck & Co., Inc.", "Merck"),
            ("LLY", "Eli Lilly and Company", "Eli Lilly"),
            # Consumer
            ("WMT", "Walmart Inc.", "Walmart"),
            ("HD", "The Home Depot, Inc.", "Home Depot"),
            ("PG", "The Procter & Gamble Company", "Procter & Gamble"),
            ("KO", "The Coca-Cola Company", "Coca-Cola"),
            ("PEP", "PepsiCo, Inc.", "PepsiCo"),
            ("COST", "Costco Wholesale Corporation", "Costco"),
            ("NKE", "NIKE, Inc.", "Nike"),
            ("MCD", "McDonald's Corporation", "McDonald's"),
            ("SBUX", "Starbucks Corporation", "Starbucks"),
            ("DIS", "The Walt Disney Company", "Disney"),
            # Energy
            ("XOM", "Exxon Mobil Corporation", "Exxon Mobil"),
            ("CVX", "Chevron Corporation", "Chevron"),
            ("COP", "ConocoPhillips", "ConocoPhillips"),
            # Telecom
            ("T", "AT&T Inc.", "AT&T"),
            ("VZ", "Verizon Communications Inc.", "Verizon"),
            # Aerospace
            ("BA", "The Boeing Company", "Boeing"),
            ("LMT", "Lockheed Martin Corporation", "Lockheed Martin"),
            # Auto
            ("F", "Ford Motor Company", "Ford"),
            ("GM", "General Motors Company", "GM"),
            # Retail
            ("TGT", "Target Corporation", "Target"),
            ("LOW", "Lowe's Companies, Inc.", "Lowe's"),
            # Pharma/Biotech
            ("MRNA", "Moderna, Inc.", "Moderna"),
            ("BNTX", "BioNTech SE", "BioNTech"),
            # Semiconductor
            ("QCOM", "QUALCOMM Incorporated", "Qualcomm"),
            ("AVGO", "Broadcom Inc.", "Broadcom"),
            ("TXN", "Texas Instruments Incorporated", "Texas Instruments"),
            # Software
            ("CRM", "Salesforce, Inc.", "Salesforce"),
            ("ORCL", "Oracle Corporation", "Oracle"),
            ("ADBE", "Adobe Inc.", "Adobe"),
            ("NOW", "ServiceNow, Inc.", "ServiceNow"),
            # Streaming/Media
            ("NFLX", "Netflix, Inc.", "Netflix"),
            ("SPOT", "Spotify Technology S.A.", "Spotify"),
            # E-commerce/Payments
            ("PYPL", "PayPal Holdings, Inc.", "PayPal"),
            ("SQ", "Block, Inc.", "Square"),
            ("SHOP", "Shopify Inc.", "Shopify"),
            # Gaming
            ("EA", "Electronic Arts Inc.", "EA"),
            ("ATVI", "Activision Blizzard, Inc.", "Activision"),
            ("TTWO", "Take-Two Interactive Software, Inc.", "Take-Two"),
            # Social Media
            ("SNAP", "Snap Inc.", "Snapchat"),
            ("PINS", "Pinterest, Inc.", "Pinterest"),
            ("TWTR", "Twitter, Inc.", "Twitter"),
            # EV/Clean Energy
            ("RIVN", "Rivian Automotive, Inc.", "Rivian"),
            ("LCID", "Lucid Group, Inc.", "Lucid"),
            ("NIO", "NIO Inc.", "NIO"),
            # Meme Stocks
            ("GME", "GameStop Corp.", "GameStop"),
            ("AMC", "AMC Entertainment Holdings, Inc.", "AMC"),
            ("BB", "BlackBerry Limited", "BlackBerry"),
            # Crypto-related
            ("COIN", "Coinbase Global, Inc.", "Coinbase"),
            ("MSTR", "MicroStrategy Incorporated", "MicroStrategy"),
            # Others
            ("UBER", "Uber Technologies, Inc.", "Uber"),
            ("LYFT", "Lyft, Inc.", "Lyft"),
            ("ABNB", "Airbnb, Inc.", "Airbnb"),
            ("DASH", "DoorDash, Inc.", "DoorDash"),
            ("ZM", "Zoom Video Communications, Inc.", "Zoom"),
            ("ROKU", "Roku, Inc.", "Roku"),
            ("SQ", "Block, Inc.", "Block"),
            # Chinese/ADR Stocks
            ("BABA", "Alibaba Group Holding Limited", "Alibaba"),
            ("JD", "JD.com, Inc.", "JD"),
            ("PDD", "PDD Holdings Inc.", "Pinduoduo"),
            ("BIDU", "Baidu, Inc.", "Baidu"),
            ("NTES", "NetEase, Inc.", "NetEase"),
            ("TME", "Tencent Music Entertainment Group", "Tencent Music"),
            ("LI", "Li Auto Inc.", "Li Auto"),
            ("XPEV", "XPeng Inc.", "XPeng"),
            # Retail
            ("DKS", "DICK'S Sporting Goods, Inc.", "Dick's Sporting Goods"),
            ("LULU", "Lululemon Athletica Inc.", "Lululemon"),
            ("GPS", "The Gap, Inc.", "Gap"),
            ("ANF", "Abercrombie & Fitch Co.", "Abercrombie"),
            ("BBWI", "Bath & Body Works, Inc.", "Bath & Body Works"),
            ("ETSY", "Etsy, Inc.", "Etsy"),
            ("CHWY", "Chewy, Inc.", "Chewy"),
            ("W", "Wayfair Inc.", "Wayfair"),
            # Airlines
            ("DAL", "Delta Air Lines, Inc.", "Delta"),
            ("UAL", "United Airlines Holdings, Inc.", "United Airlines"),
            ("AAL", "American Airlines Group Inc.", "American Airlines"),
            ("LUV", "Southwest Airlines Co.", "Southwest"),
            # Banks (more)
            ("USB", "U.S. Bancorp", "US Bank"),
            ("PNC", "The PNC Financial Services Group, Inc.", "PNC"),
            ("TFC", "Truist Financial Corporation", "Truist"),
            ("SCHW", "The Charles Schwab Corporation", "Schwab"),
            # Insurance
            ("BRK.A", "Berkshire Hathaway Inc.", "Berkshire"),
            ("BRK.B", "Berkshire Hathaway Inc.", "Berkshire"),
            ("MET", "MetLife, Inc.", "MetLife"),
            ("AIG", "American International Group, Inc.", "AIG"),
            ("PRU", "Prudential Financial, Inc.", "Prudential"),
            # Industrials
            ("CAT", "Caterpillar Inc.", "Caterpillar"),
            ("DE", "Deere & Company", "John Deere"),
            ("MMM", "3M Company", "3M"),
            ("GE", "General Electric Company", "GE"),
            ("HON", "Honeywell International Inc.", "Honeywell"),
            ("UPS", "United Parcel Service, Inc.", "UPS"),
            ("FDX", "FedEx Corporation", "FedEx"),
            # More Tech
            ("IBM", "International Business Machines Corporation", "IBM"),
            ("CSCO", "Cisco Systems, Inc.", "Cisco"),
            ("HPQ", "HP Inc.", "HP"),
            ("DELL", "Dell Technologies Inc.", "Dell"),
            ("VMW", "VMware, Inc.", "VMware"),
            ("PANW", "Palo Alto Networks, Inc.", "Palo Alto"),
            ("CRWD", "CrowdStrike Holdings, Inc.", "CrowdStrike"),
            ("ZS", "Zscaler, Inc.", "Zscaler"),
            ("SNOW", "Snowflake Inc.", "Snowflake"),
            ("PLTR", "Palantir Technologies Inc.", "Palantir"),
            ("DDOG", "Datadog, Inc.", "Datadog"),
            ("MDB", "MongoDB, Inc.", "MongoDB"),
            ("NET", "Cloudflare, Inc.", "Cloudflare"),
            ("U", "Unity Software Inc.", "Unity"),
            ("RBLX", "Roblox Corporation", "Roblox"),
            # Food & Beverage
            ("MCD", "McDonald's Corporation", "McDonald's"),
            ("YUM", "Yum! Brands, Inc.", "Yum"),
            ("CMG", "Chipotle Mexican Grill, Inc.", "Chipotle"),
            ("DPZ", "Domino's Pizza, Inc.", "Domino's"),
            ("WING", "Wingstop Inc.", "Wingstop"),
            # Cannabis
            ("TLRY", "Tilray Brands, Inc.", "Tilray"),
            ("CGC", "Canopy Growth Corporation", "Canopy"),
            # Biotech
            ("GILD", "Gilead Sciences, Inc.", "Gilead"),
            ("REGN", "Regeneron Pharmaceuticals, Inc.", "Regeneron"),
            ("VRTX", "Vertex Pharmaceuticals Incorporated", "Vertex"),
            ("BIIB", "Biogen Inc.", "Biogen"),
            ("ILMN", "Illumina, Inc.", "Illumina"),
            # REITs
            ("SPG", "Simon Property Group, Inc.", "Simon Property"),
            ("AMT", "American Tower Corporation", "American Tower"),
            ("CCI", "Crown Castle Inc.", "Crown Castle"),
            ("EQIX", "Equinix, Inc.", "Equinix"),
            # Energy (more)
            ("OXY", "Occidental Petroleum Corporation", "Occidental"),
            ("SLB", "Schlumberger Limited", "Schlumberger"),
            ("HAL", "Halliburton Company", "Halliburton"),
            ("DVN", "Devon Energy Corporation", "Devon"),
            ("MPC", "Marathon Petroleum Corporation", "Marathon"),
            ("VLO", "Valero Energy Corporation", "Valero"),
            ("PSX", "Phillips 66", "Phillips 66"),
        ]

        self._stocks = {}
        for ticker, company_name, short_name in fallback_stocks:
            self._stocks[ticker] = {
                "ticker": ticker,
                "company_name": company_name,
                "common_names": [short_name, ticker],
                "exchange": "US",
                "is_active": True,
                "source": "fallback",
            }

        logger.info(f"Built fallback database with {len(self._stocks)} common stocks")
