from collections.abc import Generator
from enum import Enum as PyEnum

from fastapi import FastAPI, Query, Depends
from pydantic import BaseModel
from datetime import datetime
from typing import Annotated, AsyncGenerator, Any
from sqlalchemy import create_engine, Integer, DateTime, Text, Enum, Engine, select
from sqlalchemy.orm import sessionmaker, Mapped, mapped_column, Session, DeclarativeBase
from sqlalchemy.sql import func
from contextlib import asynccontextmanager

from starlette import status
from starlette.requests import Request

SQLALCHEMY_DATABASE_URL = "sqlite:///./reviews.db"
POSITIVE_WORDS = ['хорош', 'люблю']
NEGATIVE_WORDS = ['плохо', 'ненавиж']


def engine_provide(
    database_url: str,
) -> Engine:
    return create_engine(
        database_url,
        connect_args={"check_same_thread": False},
    )


def engine_depend(
    request: Request,
) -> Engine:
    return request.state.engine


def session_factory_depend(
    engine: Annotated[Engine, Depends(engine_depend)],
) -> sessionmaker[Session]:
    return sessionmaker(
        autoflush=True,
        bind=engine,
        expire_on_commit=False,
    )


def session_depend(
    session_factory: Annotated[sessionmaker[Session], Depends(session_factory_depend)],
) -> Generator[Session]:
    with session_factory() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


class BaseORM(DeclarativeBase):
    pass


class SentimentEnum(PyEnum):
    positive = 'positive'
    neutral = 'neutral'
    negative = 'negative'


class ReviewORM(BaseORM):
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    sentiment: Mapped[str] = mapped_column(Enum(SentimentEnum), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
    )


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncGenerator[dict[str, Any]]:
    engine = engine_provide(SQLALCHEMY_DATABASE_URL)
    BaseORM.metadata.create_all(bind=engine)
    yield {'engine': engine}
    engine.dispose()


app = FastAPI(lifespan=lifespan)


class SentimentTypeDTO(str, PyEnum):
    positive = 'positive'
    neutral = 'neutral'
    negative = 'negative'


class ReviewCreateDTO(BaseModel):
    text: str
    sentiment: SentimentTypeDTO


class ReviewResponseDTO(
    BaseModel,
    from_attributes=True,
):
    id: int
    text: str
    sentiment: SentimentTypeDTO
    created_at: datetime


class ReviewRepository:
    def __init__(
        self,
        session: Session,
    ) -> None:
        self._session = session

    async def add_one(
        self,
        review_in: ReviewCreateDTO,
    ) -> ReviewResponseDTO:
        review = ReviewORM(
            text=review_in.text,
            sentiment=review_in.sentiment,
        )

        self._session.add(review)
        self._session.flush()
        self._session.refresh(review)
        return ReviewResponseDTO.model_validate(review)

    async def get_by_sentiment(
        self,
        sentiment: SentimentTypeDTO,
    ) -> list[ReviewResponseDTO]:
        stmt = select(ReviewORM).where(ReviewORM.sentiment == sentiment)
        result = self._session.execute(stmt).scalars().all()
        return [ReviewResponseDTO.model_validate(review) for review in result]


class SentimentTypeAPI(str, PyEnum):
    positive = 'positive'
    neutral = 'neutral'
    negative = 'negative'


class ReviewCreateAPI(BaseModel):
    text: str


class ReviewResponseAPI(
    BaseModel,
    from_attributes=True,
):
    id: int
    text: str
    sentiment: SentimentTypeAPI
    created_at: datetime


class ReviewService:
    def __init__(
        self,
        review_repository: Annotated[ReviewRepository, Depends()],
    ) -> None:
        self._review_repository = review_repository

    async def add_review(
        self,
        review_in: ReviewCreateAPI,
    ) -> ReviewResponseAPI:
        sentiment = self.analyze_sentiment(review_in.text)
        review = await self._review_repository.add_one(
            ReviewCreateDTO(
                **review_in.model_dump(),
                sentiment=SentimentTypeDTO(sentiment),
            )
        )
        return ReviewResponseAPI(**review.model_dump())

    async def get_by_sentiment(
        self,
        sentiment: SentimentTypeAPI,
    ) -> list[ReviewResponseAPI]:
        reviews = await self._review_repository.get_by_sentiment(
            sentiment=SentimentTypeDTO(sentiment),
        )
        return [ReviewResponseAPI(**review.model_dump()) for review in reviews]

    @staticmethod
    def analyze_sentiment(
        text: str,
    ) -> SentimentTypeAPI:
        text_lower = text.lower()

        if any(word in text_lower for word in POSITIVE_WORDS):
            return SentimentTypeAPI.positive
        elif any(word in text_lower for word in NEGATIVE_WORDS):
            return SentimentTypeAPI.negative
        return SentimentTypeAPI.neutral


@app.post(
    "/reviews",
    status_code=status.HTTP_201_CREATED,
    response_model=ReviewResponseAPI,
)
async def create_review(
    review_in: ReviewCreateAPI,
    review_service: Annotated[ReviewService, Depends()],
) -> ReviewResponseAPI:
    return await review_service.add_review(review_in=review_in)


@app.get(
    "/reviews",
    status_code=status.HTTP_200_OK,
    response_model=list[ReviewResponseAPI],
)
async def get_reviews(
    review_servie: Annotated[ReviewService, Depends()],
    sentiment: Annotated[SentimentTypeAPI, Query()],
) -> list[ReviewResponseAPI]:
    return await review_servie.get_by_sentiment(sentiment=sentiment)
