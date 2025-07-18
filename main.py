from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from enum import Enum as PyEnum
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Query
from pydantic import BaseModel
from sqlalchemy import DateTime, Engine, Enum, Integer, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.sql import func
from starlette import status
from starlette.requests import Request
import uvicorn

SQLALCHEMY_DATABASE_URL = "sqlite:///./reviews.db"
POSITIVE_WORDS = ['хорош', 'люблю']
NEGATIVE_WORDS = ['плохо', 'ненавиж']


@contextmanager
def engine_provide(
    database_url: str,
) -> Generator[Engine]:
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False},
    )
    yield engine
    engine.dispose()


@contextmanager
def session_factory_provide(
    engine: Engine,
) -> Generator[sessionmaker[Session]]:
    yield sessionmaker(
        autoflush=True,
        bind=engine,
        expire_on_commit=False,
    )


@contextmanager
def session_provide(
    session_factory: sessionmaker[Session],
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


class SentimentTypeDTO(str, PyEnum):
    positive = 'positive'
    neutral = 'neutral'
    negative = 'negative'


class ReviewCreateDTO(BaseModel):
    text: str
    sentiment: SentimentTypeDTO


class ReviewResponseDTO(BaseModel):
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
        return ReviewResponseDTO.model_validate(review, from_attributes=True)

    async def get_by_sentiment(
        self,
        sentiment: SentimentTypeDTO,
    ) -> list[ReviewResponseDTO]:
        stmt = select(ReviewORM).where(ReviewORM.sentiment == sentiment)
        result = self._session.execute(stmt).scalars().all()
        return [ReviewResponseDTO.model_validate(review, from_attributes=True) for review in result]


class SentimentTypeAPI(str, PyEnum):
    positive = 'positive'
    neutral = 'neutral'
    negative = 'negative'


class ReviewCreateAPI(BaseModel):
    text: str


class ReviewResponseAPI(BaseModel):
    id: int
    text: str
    sentiment: SentimentTypeAPI
    created_at: datetime


class ListReviewResponseAPI(BaseModel):
    reviews: list[ReviewResponseAPI]


class ReviewService:
    def __init__(
        self,
        review_repository: ReviewRepository,
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
    ) -> ListReviewResponseAPI:
        reviews = await self._review_repository.get_by_sentiment(
            sentiment=SentimentTypeDTO(sentiment),
        )
        return ListReviewResponseAPI(
            reviews=[ReviewResponseAPI(**review.model_dump()) for review in reviews]
        )


    @staticmethod
    def analyze_sentiment(
        text: str,
    ) -> SentimentTypeAPI:
        text_lower = text.lower()

        if any(word in text_lower for word in POSITIVE_WORDS):
            return SentimentTypeAPI.positive
        if any(word in text_lower for word in NEGATIVE_WORDS):
            return SentimentTypeAPI.negative
        return SentimentTypeAPI.neutral


def engine_depend(
    request: Request,
) -> Engine:
    return request.state.engine


def session_factory_depend(
    engine: Annotated[Engine, Depends(engine_depend)]
) -> Generator[sessionmaker[Session]]:
    with session_factory_provide(engine) as session_factory:
        yield session_factory


def session_depend(
    session_factory: Annotated[sessionmaker[Session], Depends(session_factory_depend)],
) -> Generator[Session]:
    with session_provide(session_factory) as session:
        yield session


def review_repository_depend(
    session: Annotated[Session, Depends(session_depend)],
) -> ReviewRepository:
    return ReviewRepository(session)


def review_service_depend(
    review_repository: Annotated[ReviewRepository, Depends(review_repository_depend)],
) -> ReviewService:
    return ReviewService(review_repository)


router_reviews = APIRouter(
    prefix='/reviews',
    tags=['Reviews'],
)


@router_reviews.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=ReviewResponseAPI,
)
async def create_review(
    review_in: ReviewCreateAPI,
    review_service: Annotated[ReviewService, Depends(review_service_depend)],
) -> ReviewResponseAPI:
    return await review_service.add_review(review_in=review_in)


@router_reviews.get(
    "",
    status_code=status.HTTP_200_OK,
    response_model=ListReviewResponseAPI,
)
async def get_reviews(
    review_servie: Annotated[ReviewService, Depends(review_service_depend)],
    sentiment: Annotated[SentimentTypeAPI, Query()],
) -> ListReviewResponseAPI:
    return await review_servie.get_by_sentiment(sentiment=sentiment)


@asynccontextmanager
async def lifespan(
    app: FastAPI,  # noqa: ARG001
) -> AsyncGenerator[dict[str, Any]]:
    with engine_provide(SQLALCHEMY_DATABASE_URL) as engine:
        BaseORM.metadata.create_all(bind=engine)
        yield {'engine': engine}


app = FastAPI(lifespan=lifespan)
app.include_router(router_reviews)

if __name__ == '__main__':
    uvicorn.run('main:app', host='localhost', port=80, reload=True)
